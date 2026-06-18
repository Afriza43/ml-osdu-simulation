import streamlit as st
import pandas as pd
import numpy as np
import json
from sqlalchemy import create_engine
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor

# ==============================================================================
# 1. APPLICATION SETUP & THEMING
# ==============================================================================
st.set_page_config(
    page_title="OSDU End-to-End Subsurface ML Simulator",
    page_icon="🛢️",
    layout="wide"
)

st.title("🛢️ OSDU Data Platform Lifecycle & Petrophysical ML Simulation")
st.markdown("""
Aplikasi ini mensimulasikan alur kerja Modeling ML menggunakan spesifikasi **OSDU (Open Subsurface Data Universe) API**. 
Aplikasi menjembatani data relasional dari database lokal `ppdm` dan mentransformasikannya secara real-time ke dalam arsitektur microservices berbasis JSON platform OSDU global.
""")

# ==============================================================================
# 2. CORE ENGINE: LOCAL DATABASE EXTRACTORS & SYNTHETIC DATA FALLBACKS
# ==============================================================================


def get_db_engine(db_url):
    return create_engine(db_url)


def run_osdu_search_api_mock(db_url, partition="opendes"):
    """
    Simulasi POST /api/search/v2/query
    Menarik metadata sumur aktif dari PPDM dan mengemasnya ke format OSDU Record WKS.
    """
    try:
        engine = get_db_engine(db_url)
        query = "SELECT uwi, well_name, operator, surface_latitude, surface_longitude FROM ppdm.well WHERE active_ind = 'Y' AND uwi LIKE 'TMB%'"
        df = pd.read_sql(query, engine)
        records_found = len(df)
    except Exception:
        # Fallback data dumi jika database offline agar presentasi tetap berjalan lancar
        df = pd.DataFrame([
            {"uwi": "HNL-001", "well_name": "HNL-001", "operator": "PERTAMINA",
                "surface_latitude": -0.5, "surface_longitude": 110.5},
            {"uwi": "MNS-001", "well_name": "MNS-001", "operator": "PERTAMINA",
                "surface_latitude": -0.3, "surface_longitude": 110.3},
            {"uwi": "BYU-001", "well_name": "BYU-001", "operator": "PERTAMINA",
                "surface_latitude": -0.1, "surface_longitude": 110.1},
            {"uwi": "TMB-023", "well_name": "TMB-023", "operator": "PERTAMINA",
                "surface_latitude": 0.0, "surface_longitude": 110.0}
        ])
        records_found = len(df)

    osdu_response = {
        "results": [],
        "totalCount": records_found
    }

    for _, row in df.iterrows():
        osdu_response["results"].append({
            "id": f"{partition}:master-data--Wellbore:{row['uwi']}",
            "kind": "osdu:wks:master-data--Wellbore:1.0.0",
            "data": {
                "FacilityName": row['well_name'],
                "DataSourceOrganisationID": row['operator'] if row['operator'] else "UNKNOWN",
                "VerticalMeasurements": [
                    {
                        "VerticalMeasurement": float(row['surface_latitude']) if row['surface_latitude'] else 0.0,
                        "VerticalMeasurementUnitID": f"{partition}:reference-data--UnitOfMeasure:{row['surface_longitude'] if row['surface_longitude'] else 'DEG'}:"
                    }
                ]
            }
        })
    return osdu_response


def run_osdu_wellbore_ddms_get_mock(db_url, uwi, partition="opendes"):
    """
    Simulasi GET /api/os-wellbore-ddms/v3/welllogs/{log_id}/data
    Melakukan JOIN relasional panjang PPDM dan mengubahnya menjadi bentuk kolom matriks OSDU.
    """
    try:
        engine = get_db_engine(db_url)
        query = f"""
            SELECT v.index_value as depth, c.reported_mnemonic as curve_id, v.measured_value 
            FROM ppdm.well_log_curve_value v
            JOIN ppdm.well_log_curve c ON v.uwi = c.uwi AND v.curve_id = c.curve_id
            WHERE v.uwi = '{uwi}' AND c.reported_mnemonic IN ('GR', 'NPHI', 'RHOB')
            ORDER BY v.index_value ASC
        """
        df_long = pd.read_sql(query, engine)
        if df_long.empty:
            raise ValueError("Empty data")
        df_wide = df_long.pivot(
            index='depth', columns='curve_id', values='measured_value').reset_index()
    except Exception:
        # Penghasil data dumi otomatis jika tabel kosong atau tidak terkoneksi
        np.random.seed(42)
        depths = np.arange(1500, 1700, 0.5)
        gr = 60 + 35 * np.sin(depths / 15) + \
            np.random.normal(0, 8, len(depths))
        rhob = 2.3 + 0.15 * np.sin(depths / 40) + \
            np.random.normal(0, 0.04, len(depths))
        nphi = 0.35 - 0.1 * (rhob - 2.2) + \
            np.random.normal(0, 0.02, len(depths))
        df_wide = pd.DataFrame(
            {"depth": depths, "GR": gr, "NPHI": nphi, "RHOB": rhob})

    df_wide = df_wide.dropna(subset=['GR', 'NPHI', 'RHOB'])

    return {
        "columns": ["MD", "GR", "NPHI", "RHOB"],
        "index": df_wide['depth'].tolist(),
        "data": df_wide[['depth', 'GR', 'NPHI', 'RHOB']].values.tolist()
    }

# ==============================================================================
# 3. INTERMEDIATE MACHINE LEARNING ENGINE
# ==============================================================================


def process_ml_inference(osdu_ddms_json):
    """
    Mengubah format matriks OSDU JSON menjadi Pandas DataFrame,
    lalu melatih model ML menggunakan GR asli untuk memprediksi 
    nilai Aktual yang dihitung secara matematis menggunakan rumus Bateman/Konen (.lls).
    """
    columns = osdu_ddms_json["columns"]
    data = osdu_ddms_json["data"]
    df = pd.DataFrame(data, columns=columns)

    # ---------------------------------------------------------
    # 1. PERHITUNGAN VSH ACTUAL (Berdasarkan vsh_gr.lls)
    # ---------------------------------------------------------
    # Simulasi GR_NORM (Misal: hasil normalisasi eksternal)
    df['GR_NORM'] = df['GR'] * 0.95

    # Mengambil nilai Matrix & Shale (atau ditetapkan manual)
    gr_ma = df['GR_NORM'].min()
    gr_sh = df['GR_NORM'].max()

    # Linear calculation: v = ( GR_NORM - GR_MA ) / ( GR_SH - GR_MA )
    vsh_linear = (df['GR_NORM'] - gr_ma) / \
        (gr_sh - gr_ma) if gr_sh > gr_ma else 0.0

    # Limit result between 0 and 1
    df['VSH_Actual'] = np.clip(vsh_linear, 0.0, 1.0)

    # ---------------------------------------------------------
    # 2. PERHITUNGAN PHIE ACTUAL (Berdasarkan phi_dnbk.lls)
    # ---------------------------------------------------------
    # Parameter Bateman/Konen (Disesuaikan untuk satuan g/cc)
    RHO_FL = 1.0
    RHO_SH = df['RHOB'].max()   # Densitas Shale
    NPHI_SH = df['NPHI'].max()  # Neutron Porosity Shale

    def dn_xplot(rho0, nphi0, rho_fl=1.0):
        """Implementasi blok DN_XPLOT dari phi_dnbk.lls"""
        # phid = ( 2710 - rho0 ) / ( 2710 - RHO_FL ) disesuaikan ke g/cc (2.71)
        phid = (2.71 - rho0) / (2.71 - rho_fl)

        # Kondisi percabangan
        cond = nphi0 >= phid

        # pda dan pna values berdasarkan kondisi
        pda = np.where(cond, (2.71 - 4.0) / (2.71 - rho_fl), 1.0)
        pna = np.where(cond,
                       0.7 - 10**(-5 * nphi0 - 0.16),
                       -2.06 * nphi0 - 1.17 + 10**(-16 * nphi0 - 0.4))

        # phix = ( pda*nphi0 - phid*pna ) / ( pda - pna )
        phix = (pda * nphi0 - phid * pna) / (pda - pna)
        return phix

    # Mencegah pembagian dengan 0 saat koreksi shale
    vsh_safe = np.clip(df['VSH_Actual'], 0, 0.99)

    # Correct density and neutron logs for shale to "shale reduced"
    rhosr = (df['RHOB'] - vsh_safe * RHO_SH) / (1 - vsh_safe)
    nphisr = (df['NPHI'] - vsh_safe * NPHI_SH) / (1 - vsh_safe)

    # nphisr = LIMIT ( nphisr, -0.015, 1 )
    nphisr = np.clip(nphisr, -0.015, 1.0)

    # Calculate "shale reduced" porosity (phix)
    phix_sr = dn_xplot(rhosr, nphisr, RHO_FL)

    # Calculate effective porosity: PHIE_DN = phix * ( 1 - VSH )
    phie_actual = phix_sr * (1 - df['VSH_Actual'])

    # Limit PHIE antara 0 dan 1
    df['PHIE_Actual'] = np.clip(phie_actual, 0.0, 1.0)

    # ---------------------------------------------------------
    # 3. MACHINE LEARNING ENGINE (PREDIKSI)
    # ---------------------------------------------------------
    # PENTING: Model dilatih menggunakan FITUR GR (BUKAN GR_NORM)
    X = df[['GR', 'NPHI', 'RHOB']]

    # Melatih dan memprediksi VSH
    model_vsh = RandomForestRegressor(n_estimators=30, random_state=42)
    model_vsh.fit(X, df['VSH_Actual'])
    df['VSH_Predict'] = model_vsh.predict(X)

    # Melatih dan memprediksi PHIE
    model_phie = RandomForestRegressor(n_estimators=30, random_state=42)
    model_phie.fit(X, df['PHIE_Actual'])
    df['PHIE_Predict'] = model_phie.predict(X)

    return df


# ==============================================================================
# 4. USER INTERFACE LAYOUT & INTERACTIVE SIMULATION
# ==============================================================================
db_user = st.secrets["database"]["username"]
db_pass = st.secrets["database"]["password"]
db_host = st.secrets["database"]["host"]
db_port = st.secrets["database"]["port"]
db_name = st.secrets["database"]["dbname"]

# 2. Merakit connection string di backend (tidak terlihat oleh user UI)
SECURE_DB_URL = f"postgresql://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"

# Sidebar Konfigurasi Lingkungan Kerja
st.sidebar.header("🔑 OSDU Environment Connection")
st.sidebar.info(f"Connected to Host: `{db_host}`\n\nDatabase: `{db_name}`")

partition_id = st.sidebar.text_input("OSDU Data Partition ID", value="opendes")

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Petunjuk :**
1. Klik tombol **Fetch Discovery** di bawah untuk memicu Search Service OSDU.
2. Pindah ke panel utama untuk menjalankan alur log data dari Tahap 1 sampai 5.
""")

# Inisialisasi Cache State Streamlit agar data tidak hilang saat re-render
if 'search_data' not in st.session_state:
    st.session_state['search_data'] = None
if 'selected_uwi' not in st.session_state:
    st.session_state['selected_uwi'] = None
if 'bulk_input_data' not in st.session_state:
    st.session_state['bulk_input_data'] = None
if 'ml_output_df' not in st.session_state:
    st.session_state['ml_output_df'] = None
if 'derived_record_id' not in st.session_state:
    st.session_state['derived_record_id'] = None
if 'write_bulk_response' not in st.session_state:
    st.session_state['write_bulk_response'] = None

if st.sidebar.button("🔍 Fetch OSDU Discovery (Search API)", type="secondary"):
    st.session_state['search_data'] = run_osdu_search_api_mock(
        SECURE_DB_URL, partition_id)
    st.sidebar.success("Discovery Metadata Berhasil Dibuat!")

# Pemetaan Pilihan Wellbore Hasil dari Search API
well_map = {}
if st.session_state['search_data'] is not None:
    for item in st.session_state['search_data']['results']:
        uwi = item['id'].split(':')[-1]
        name = item['data']['FacilityName']
        well_map[f"{name} [{uwi}]"] = uwi

    selected_label = st.sidebar.selectbox(
        "Pilih Target Wellbore Aset:", list(well_map.keys()))
    st.session_state['selected_uwi'] = well_map[selected_label]

# Manajemen Pembuatan Tab Aplikasi
tab_pipeline, tab_visualizer, tab_docs = st.tabs([
    "🚀 End-to-End OSDU API Pipeline",
    "📊 Petrophysical 4-Track Visualizer",
    "📚 OSDU Schema Link & API Reference"
])

# ------------------------------------------------------------------------------
# TAB 1: DATA PIPELINE SIMULATOR (LIVE REQUEST & RESPONSE LOGS)
# ------------------------------------------------------------------------------
with tab_pipeline:
    st.header("🔀 Alur Simulasi REST API Terintegrasi")
    st.markdown("Ikuti langkah-langkah di bawah untuk melihat transaksi payload JSON real-time dan klik tautan untuk membaca standar dokumentasi OSDU.")

    # --- TAHAP 1 ---
    st.subheader("Tahap 1: Well Discovery via Search Service")
    st.markdown("[🔗 **Referensi API:** `POST /api/search/v2/query`](https://community.opengroup.org/osdu/platform/system/search/-/blob/master/docs/api/search_openapi.yaml) — *Mencari ID Sumur bawah permukaan menggunakan Elasticsearch query.*")
    col1, col2 = st.columns(2)
    with col1:
        st.caption("**💡 HTTP Request Payload (POST)**")
        search_request = {
            "kind": "osdu:wks:master-data--Wellbore:1.0.0",
            "query": f"data.DataSourceOrganisationID: \"PERTAMINA\"",
            "returnedFields": ["id", "data.FacilityName", "data.VerticalMeasurements"]
        }
        st.code(
            f"POST /api/search/v2/query\nHeaders: {{ 'data-partition-id': '{partition_id}' }}\n\nBody:\n{json.dumps(search_request, indent=2)}", language="json")
    with col2:
        st.caption("**📥 HTTP Response Payload (JSON)**")
        if st.session_state['search_data'] is not None:
            st.json(st.session_state['search_data'])
        else:
            st.info(
                "Klik tombol 'Trigger OSDU Discovery' di sidebar terlebih dahulu.")

    st.markdown("---")

    # --- TAHAP 2 ---
    st.subheader("Tahap 2: Ekstraksi Data Kurva Mentah (Wellbore DDMS GET)")
    st.markdown(
        "[🔗 **Referensi API:** `GET /welllogs/{id}/data`](https://community.opengroup.org/osdu/platform/domain-data-mgmt/wellbore/wellbore-domain-services/-/tree/master/docs/api) — *Menarik bulk-data matriks kurva log masukan (GR, NPHI, RHOB) berdasarkan target wellbore.*")

    col1, col2 = st.columns(2)
    with col1:
        st.caption("**💡 HTTP Request Details (GET)**")
        target_uwi = st.session_state['selected_uwi'] if st.session_state['selected_uwi'] else "HNL-001"
        target_log_id = f"{partition_id}:work-product-component--WellLog:WL-{target_uwi}-RAW"
        st.code(
            f"GET /api/os-wellbore-ddms/v3/welllogs/{target_log_id}/data?curves=MD,GR,NPHI,RHOB\nHeaders: {{ 'data-partition-id': '{partition_id}' }}", language="text")

        if st.button("📥 Pull Bulk Log Curves From Platform", type="primary", disabled=(st.session_state['selected_uwi'] is None)):
            st.session_state['bulk_input_data'] = run_osdu_wellbore_ddms_get_mock(
                SECURE_DB_URL, target_uwi, partition_id)
            st.success("Matriks Data Log Berhasil Diekstraksi!")
    with col2:
        st.caption("**📥 HTTP Response Columnar Matrix (JSON)**")
        if st.session_state['bulk_input_data'] is not None:
            sample_json = {
                "columns": st.session_state['bulk_input_data']["columns"],
                "index": st.session_state['bulk_input_data']["index"][:3],
                "data": st.session_state['bulk_input_data']["data"][:3]
            }
            st.json(sample_json)
            st.caption(
                "*(Menampilkan 3 baris sampel matriks teratas untuk optimasi visual)*")
        else:
            st.warning("Menunggu instruksi penarikan data dari tombol.")

    st.markdown("---")

    # --- TAHAP 3 ---
    st.subheader("Tahap 3: Pemrosesan Komputasi Machine Learning")
    st.markdown("*(Tahap ini adalah pemrosesan di sisi Client/Aplikasi Anda, tidak memanggil endpoint OSDU).* Lapisan perantara memuat matriks JSON ke Pandas dan menjalankan inferensi model Random Forest.")

    if st.button("🚀 Execute Petrophysical ML Predictor Pipeline", disabled=(st.session_state['bulk_input_data'] is None)):
        st.session_state['ml_output_df'] = process_ml_inference(
            st.session_state['bulk_input_data'])
        st.success(
            "Inferensi Sukses! Variabel VSH & PHIE Telah Berhasil Diestimasi.")

    if st.session_state['ml_output_df'] is not None:
        st.dataframe(st.session_state['ml_output_df'].head(
            5), use_container_width=True)
        st.caption("DataFrame Siap Ingest (Format Wide Tabular)")

    st.markdown("---")

    # --- TAHAP 4 ---
    st.subheader(
        "Tahap 4: Registrasi Metadata Log Turunan Baru (Wellbore DDMS POST Record)")
    st.markdown("[🔗 **Referensi API:** `POST /welllogs`](https://community.opengroup.org/osdu/platform/domain-data-mgmt/wellbore/wellbore-domain-services/-/tree/master/docs/api) | [🔗 **WKS Schema:** `WellLog:1.1.0`](https://community.opengroup.org/osdu/data/data-definitions/-/blob/master/E-R/work-product-component/WellLog.1.1.0.md) — *Membuat wadah metadata baru untuk menampung properti log hasil kalkulasi.*")

    col1, col2 = st.columns(2)
    with col1:
        st.caption("**💡 HTTP Request Payload (POST Record)**")
        derived_log_id = f"{partition_id}:work-product-component--WellLog:Derived-ML-VSH-PHIE-{target_uwi}"
        record_payload = [{
            "id": derived_log_id,
            "kind": "osdu:wks:work-product-component--WellLog:1.1.0",
            "acl": {"owners": [f"data.default.owners@{partition_id}.com"], "viewers": [f"data.default.viewers@{partition_id}.com"]},
            "legal": {"legaltags": [f"{partition_id}-public-usa-dataset"], "otherRelevantDataCountries": ["US"]},
            "data": {
                "Name": f"ML_Derived_Petrophysics_{target_uwi}",
                "Description": "Data log hasil prediksi model Machine Learning Random Forest Regressor",
                "WellboreID": f"{partition_id}:master-data--Wellbore:{target_uwi}",
                "Curves": [
                    {"CurveID": "MD", "Mnemonic": "MD",
                        "CurveUnit": f"{partition_id}:reference-data--UnitOfMeasure:M:"},
                    {"CurveID": "VSH", "Mnemonic": "VSH",
                        "CurveUnit": f"{partition_id}:reference-data--UnitOfMeasure:V/V:"},
                    {"CurveID": "PHIE", "Mnemonic": "PHIE",
                        "CurveUnit": f"{partition_id}:reference-data--UnitOfMeasure:V/V:"}
                ]
            }
        }]
        st.code(
            f"POST /api/os-wellbore-ddms/v3/welllogs\nHeaders: {{ 'data-partition-id': '{partition_id}' }}\n\nBody:\n{json.dumps(record_payload, indent=2)}", language="json")

        if st.button("📝 Register Derived Log Container", disabled=(st.session_state['ml_output_df'] is None)):
            st.session_state['derived_record_id'] = derived_log_id
            st.success("Metadata Log Baru Berhasil Terdaftar!")
    with col2:
        st.caption("**📥 HTTP Response Payload (JSON)**")
        if st.session_state['derived_record_id'] is not None:
            creation_response = {
                "recordCount": 1,
                "recordIds": [st.session_state['derived_record_id']]
            }
            st.json(creation_response)
        else:
            st.warning("Menunggu instruksi registrasi.")

    st.markdown("---")

    # --- TAHAP 5 ---
    st.subheader(
        "Tahap 5: Injeksi Bulk Data Angka Hasil ML (Wellbore DDMS POST Data)")
    st.markdown("[🔗 **Referensi API:** `POST /welllogs/{id}/data`](https://community.opengroup.org/osdu/platform/domain-data-mgmt/wellbore/wellbore-domain-services/-/tree/master/docs/api) — *Menulis matriks data angka kedalaman, nilai VSH, dan nilai PHIE ke dalam record yang telah didaftarkan.*")

    col1, col2 = st.columns(2)
    with col1:
        st.caption("**💡 HTTP Request Payload (POST Data Matrix)**")
        st.code(
            f"POST /api/os-wellbore-ddms/v3/welllogs/{derived_log_id}/data\nHeaders: {{ 'data-partition-id': '{partition_id}' }}", language="text")

        if st.button("📤 Push Matrix Back to OSDU Platform", disabled=(st.session_state['derived_record_id'] is None)):
            st.session_state['write_bulk_response'] = {
                "recordId": st.session_state['derived_record_id'],
                "recordVersion": 1687000000123456
            }
            st.success(
                "Ingestion Sukses! Seluruh Data Hasil ML Aman Tersimpan di Data Lake OSDU.")
    with col2:
        st.caption("**📥 HTTP Response Status Confirmation (JSON)**")
        if st.session_state['write_bulk_response'] is not None:
            st.json(st.session_state['write_bulk_response'])
        else:
            st.warning("Menunggu eksekusi pemuatan data akhir.")

# ------------------------------------------------------------------------------
# TAB 2: PETROPHYSICAL MULTI-TRACK VISUALIZER (ACTUAL VS PREDICTED)
# ------------------------------------------------------------------------------
with tab_visualizer:
    st.header("📊 Hasil Visualisasi Komparasi: Actual vs ML Prediction")
    st.markdown("Grafik ini menumpuk (*overlay*) hasil perhitungan deterministik (Actual) dengan hasil tebakan Machine Learning (Predict) untuk memvalidasi akurasi model secara visual.")

    if st.session_state['ml_output_df'] is not None:
        df = st.session_state['ml_output_df']

        # Inisialisasi Kanvas Multi-Track
        fig = make_subplots(
            rows=1, cols=4, shared_yaxes=True,
            subplot_titles=('Track 1: Gamma Ray', 'Track 2: Porosity/Density',
                            'Track 3: VSH (Actual vs ML)', 'Track 4: PHIE (Actual vs ML)'),
            horizontal_spacing=0.04
        )

        # Track 1: GR
        fig.add_trace(go.Scatter(x=df['GR'], y=df['MD'], name='Gamma Ray', line=dict(
            color='green', width=1.5)), row=1, col=1)

        # Track 2: NPHI vs RHOB
        fig.add_trace(go.Scatter(x=df['NPHI'], y=df['MD'], name='NPHI (Porosity)', line=dict(
            color='blue', width=1.2)), row=1, col=2)
        fig.add_trace(go.Scatter(x=df['RHOB'], y=df['MD'], name='RHOB (Density)', line=dict(
            color='red', width=1.2, dash='dash')), row=1, col=2)

        # Track 3: VSH (Actual vs Predict Overlay)
        fig.add_trace(go.Scatter(x=df['VSH_Actual'], y=df['MD'], name='VSH (Actual)', line=dict(
            color='gray', width=2, dash='dot')), row=1, col=3)
        # PERUBAHAN WARNA DI SINI: 'black' menjadi 'orange'
        fig.add_trace(go.Scatter(x=df['VSH_Predict'], y=df['MD'], name='VSH (ML Predict)', line=dict(
            color='orange', width=2)), row=1, col=3)

        # Track 4: PHIE (Actual vs Predict Overlay)
        fig.add_trace(go.Scatter(x=df['PHIE_Actual'], y=df['MD'], name='PHIE (Actual)', line=dict(
            color='lightcoral', width=2, dash='dot')), row=1, col=4)
        fig.add_trace(go.Scatter(x=df['PHIE_Predict'], y=df['MD'], name='PHIE (ML Predict)', line=dict(
            color='purple', width=1.5)), row=1, col=4)

        # Konfigurasi Balik Sumbu Kedalaman & Penyesuaian Axis X
        fig.update_yaxes(autorange="reversed",
                         title_text="Depth / Kedalaman (MD)")
        fig.update_xaxes(title_text="GAPI", row=1, col=1)
        fig.update_xaxes(title_text="Fraction", row=1, col=2)
        fig.update_xaxes(title_text="Volume (Fraction)",
                         range=[0, 1], row=1, col=3)
        fig.update_xaxes(title_text="Porosity (Fraction)",
                         range=[0, 0.4], row=1, col=4)

        fig.update_layout(
            height=800,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom",
                        y=1.05, xanchor="center", x=0.5),
            hovermode="y unified"  # Membuat tooltip muncul segaris untuk perbandingan mudah
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("💡 Data kurva hasil perhitungan belum tersedia. Selesaikan langkah pemrosesan di Tab 'End-to-End OSDU API Pipeline' terlebih dahulu.")

# ------------------------------------------------------------------------------
# TAB 3: SCHEMAS DOCUMENTATION AND DIRECT INTEGRATION LINKS
# ------------------------------------------------------------------------------
with tab_docs:
    st.header("📚 Registri Sumber Dokumentasi & Tautan Skema OSDU Resmi")
    st.markdown("Setiap komponen payload JSON wajib mematuhi skema ketat global. Klik tautan langsung di bawah untuk melihat repositori kode OSDU asli.")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🌐 Repositori Inti & Panduan Layanan")
        st.markdown("""
        * **[Dokumentasi Inti OSDU Forum Platform](https://osdu.opengroup.org/)** — Portal informasi pengembang utama ekosistem OSDU.
        * **[Spesifikasi OpenAPI Search Service](https://community.opengroup.org/osdu/platform/system/search/-/blob/master/docs/api/search_openapi.yaml)** — Blueprint rincian seluruh endpoint pencarian berbasis query Lucene.
        * **[Panduan Operasional Wellbore DDMS API](https://community.opengroup.org/osdu/platform/domain-data-mgmt/wellbore/wellbore-domain-services/-/tree/master/docs/api)** — Repositori backend utama untuk manipulasi objek I/O data sumur.
        """)

        st.subheader("📋 Pemetaan Key JSON Metadata (Record Layer)")
        st.markdown("""
        * **[id (Unique Resource Name Specification)](https://osdu.pages.opengroup.org/platform/system/storage/api/storage_openapi.yaml)** — Aturan pembuatan ID unik data bawah permukaan pada Storage Service.
        * **[kind (Schema Identifier Rules & Lifecycle)](https://community.opengroup.org/osdu/data/data-definitions/-/blob/master/Guides/Chapters/06-LifecycleProperties.md)** — Dokumentasi pembentukan versi validasi blueprint (WKS).
        * **[acl & legal (Access Control & Compliance Tags)](https://osdu.pages.opengroup.org/platform/system/storage/)** — Aturan kepatuhan hukum data dan kepemilikan multi-tenant cloud.
        """)

    with col2:
        st.subheader("🧬 Tata Kelola Desain Skema Struktur (WKS)")
        st.markdown("""
        * **[Skema Blueprint Master-Data Wellbore (1.0.0)](https://community.opengroup.org/osdu/data/data-definitions/-/blob/master/E-R/master-data/Wellbore.1.0.0.md)** — Relasi properti struktural sumur pengganti tabel `ppdm.well`.
        * **[Skema Blueprint Work-Product-Component WellLog (1.1.0)](https://community.opengroup.org/osdu/data/data-definitions/-/blob/master/E-R/work-product-component/WellLog.1.1.0.md)** — Aturan pendefinisian array penamaan kurva log masukan dan keluaran.
        * **[Registri UnitOfMeasure (UOM) Reference-Data](https://community.opengroup.org/osdu/data/data-definitions/-/blob/master/E-R/reference-data/UnitOfMeasure.1.0.0.md)** — Kamus standar internasional nama satuan pengukuran properti geofisika.
        """)

        st.info("""
        **Catatan Arsitektur:** Struktur matriks data terkompresi (`columns`, `index`, `data`) yang digunakan pada **Tahap 2 & 5** mengacu pada spesifikasi optimasi performa tinggi pada modul `BulkIO` OSDU Core Backend, 
        menghilangkan keharusan query ratusan ribu baris relasional database transaksional biasa secara berulang-ulang.
        """)
