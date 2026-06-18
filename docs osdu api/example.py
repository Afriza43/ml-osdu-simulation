import asyncio
from pprint import pprint

from odes_schema import AsyncApis as schemaApi, ApiClient, AuthApiClient
from odes_schema.models import SchemaInfoResponse

HOSTNAME = "http://127.0.0.1:8080"
TOKEN = "Bearer eyJ*******"
DATA_PARTITION = "my_data_partition"


def make_schema_api_with_middleware():
    async def client_middleware(request, call_next):    
        request.headers["Authorization"] = TOKEN
        response = await call_next(request)
        return response

    api_client = ApiClient(host=HOSTNAME)
    api_client.add_middleware(middleware=client_middleware)
    return schemaApi(api_client)


def make_schema_api_with_auth_token():    
    return schemaApi(AuthApiClient(host=HOSTNAME, token=TOKEN))


async def main():
     
    if 1:
        schema_api = make_schema_api_with_middleware()
    else:
        schema_api = make_schema_api_with_auth_token()

    kind_id = "osdu:wks:work-product-component--WellLog:1.1.0"
    splitted_id = kind_id.split(":")
    version = splitted_id[3].split(".")
    get_schema_response = await schema_api.schema_api.get_schema(
        data_partition_id=DATA_PARTITION, id=kind_id
    )

    schema_info_response: SchemaInfoResponse = await schema_api.schema_api.search_schema_info_repository(
        data_partition_id=DATA_PARTITION,
        authority=splitted_id[0],
        source=splitted_id[1],
        entity_type=splitted_id[2],
        schema_version_major=version[0],
        schema_version_minor=version[1],
        schema_version_patch=version[2],
    )
   
    assert isinstance(schema_info_response, SchemaInfoResponse)
    assert schema_info_response.schema_infos[0].schema_identity.id == kind_id

    pprint(get_schema_response)

    await schema_api.client.close()


if __name__ == "__main__":
    asyncio.run(main())
