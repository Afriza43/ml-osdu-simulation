# flake8: noqa E501
from asyncio import get_event_loop
from typing import TYPE_CHECKING, Awaitable

from fastapi.encoders import jsonable_encoder

from odes_search import models as m

if TYPE_CHECKING:
    from odes_search.api_client import ApiClient


class _SearchApi:
    def __init__(self, api_client: "ApiClient"):
        self.api_client = api_client

    def _build_for_c_cs_query(
        self, data_partition_id: str, ccs_query_request: m.CcsQueryRequest = None
    ) -> Awaitable[m.CcsQueryResponse]:
        """
        The API supports cross cluster searches when given the list of partitions.
        """
        headers = {"data-partition-id": str(data_partition_id)}

        body = jsonable_encoder(ccs_query_request)

        return self.api_client.request(
            type_=m.CcsQueryResponse, method="POST", url="/v2/ccs/query", headers=headers, json=body
        )

    def _build_for_delete_index(self, kind: str, data_partition_id: str) -> Awaitable[None]:
        """
        The API can be used to purge all indexed documents for a kind. Required roles: 'users.datalake.admins' or 'users.datalake.ops'
        """
        path_params = {"kind": str(kind)}

        headers = {"data-partition-id": str(data_partition_id)}

        return self.api_client.request(
            type_=None,
            method="DELETE",
            url="/v2/index/{kind}",
            path_params=path_params,
            headers=headers,
        )

    def _build_for_get_index_schema(self, kind: str, data_partition_id: str) -> Awaitable[str]:
        """
        The API returns the schema for a given kind which is used find what attributes are indexed and their respective data types (at index time). Required roles: 'users.datalake.viewers' or 'users.datalake.editors' or 'users.datalake.admins' or 'users.datalake.ops'
        """
        path_params = {"kind": str(kind)}

        headers = {"data-partition-id": str(data_partition_id)}

        return self.api_client.request(
            type_=str,
            method="GET",
            url="/v2/index/schema/{kind}",
            path_params=path_params,
            headers=headers,
        )

    def _build_for_query(
        self, data_partition_id: str, query_request: m.QueryRequest = None
    ) -> Awaitable[m.QueryResponse]:
        """
        The API supports full text search on string fields, range queries on date, numeric or string fields, along with geo-spatial search. Required roles: 'users.datalake.viewers' or 'users.datalake.editors' or 'users.datalake.admins' or 'users.datalake.ops'. In addition, users must be a member of data groups to access the data.
        """
        headers = {"data-partition-id": str(data_partition_id)}

        body = jsonable_encoder(query_request)

        return self.api_client.request(
            type_=m.QueryResponse, method="POST", url="/v2/query", headers=headers, json=body
        )

    def _build_for_query_with_cursor(
        self, data_partition_id: str, cursor_query_request: m.CursorQueryRequest = None
    ) -> Awaitable[m.CursorQueryResponse]:
        """
        The API supports full text search on string fields, range queries on date, numeric or string fields, along with geo-spatial search. Required roles: 'users.datalake.viewers' or 'users.datalake.editors' or 'users.datalake.admins' or 'users.datalake.ops'. In addition, users must be a member of data groups to access the data. It can be used to retrieve large numbers of results (or even all results) from a single search request, in much the same way as you would use a cursor on a traditional database.
        """
        headers = {"data-partition-id": str(data_partition_id)}

        body = jsonable_encoder(cursor_query_request)

        return self.api_client.request(
            type_=m.CursorQueryResponse, method="POST", url="/v2/query_with_cursor", headers=headers, json=body
        )


class AsyncSearchApi(_SearchApi):
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.api_client.close()

    async def c_cs_query(
        self, data_partition_id: str, ccs_query_request: m.CcsQueryRequest = None
    ) -> m.CcsQueryResponse:
        """
        The API supports cross cluster searches when given the list of partitions.
        """
        return await self._build_for_c_cs_query(
            data_partition_id=data_partition_id, ccs_query_request=ccs_query_request
        )

    async def delete_index(self, kind: str, data_partition_id: str) -> None:
        """
        The API can be used to purge all indexed documents for a kind. Required roles: 'users.datalake.admins' or 'users.datalake.ops'
        """
        return await self._build_for_delete_index(kind=kind, data_partition_id=data_partition_id)

    async def get_index_schema(self, kind: str, data_partition_id: str) -> str:
        """
        The API returns the schema for a given kind which is used find what attributes are indexed and their respective data types (at index time). Required roles: 'users.datalake.viewers' or 'users.datalake.editors' or 'users.datalake.admins' or 'users.datalake.ops'
        """
        return await self._build_for_get_index_schema(kind=kind, data_partition_id=data_partition_id)

    async def query(self, data_partition_id: str, query_request: m.QueryRequest = None) -> m.QueryResponse:
        """
        The API supports full text search on string fields, range queries on date, numeric or string fields, along with geo-spatial search. Required roles: 'users.datalake.viewers' or 'users.datalake.editors' or 'users.datalake.admins' or 'users.datalake.ops'. In addition, users must be a member of data groups to access the data.
        """
        return await self._build_for_query(data_partition_id=data_partition_id, query_request=query_request)

    async def query_with_cursor(
        self, data_partition_id: str, cursor_query_request: m.CursorQueryRequest = None
    ) -> m.CursorQueryResponse:
        """
        The API supports full text search on string fields, range queries on date, numeric or string fields, along with geo-spatial search. Required roles: 'users.datalake.viewers' or 'users.datalake.editors' or 'users.datalake.admins' or 'users.datalake.ops'. In addition, users must be a member of data groups to access the data. It can be used to retrieve large numbers of results (or even all results) from a single search request, in much the same way as you would use a cursor on a traditional database.
        """
        return await self._build_for_query_with_cursor(
            data_partition_id=data_partition_id, cursor_query_request=cursor_query_request
        )


class SyncSearchApi(_SearchApi):
    def c_cs_query(self, data_partition_id: str, ccs_query_request: m.CcsQueryRequest = None) -> m.CcsQueryResponse:
        """
        The API supports cross cluster searches when given the list of partitions.
        """
        coroutine = self._build_for_c_cs_query(data_partition_id=data_partition_id, ccs_query_request=ccs_query_request)
        return get_event_loop().run_until_complete(coroutine)

    def delete_index(self, kind: str, data_partition_id: str) -> None:
        """
        The API can be used to purge all indexed documents for a kind. Required roles: 'users.datalake.admins' or 'users.datalake.ops'
        """
        coroutine = self._build_for_delete_index(kind=kind, data_partition_id=data_partition_id)
        return get_event_loop().run_until_complete(coroutine)

    def get_index_schema(self, kind: str, data_partition_id: str) -> str:
        """
        The API returns the schema for a given kind which is used find what attributes are indexed and their respective data types (at index time). Required roles: 'users.datalake.viewers' or 'users.datalake.editors' or 'users.datalake.admins' or 'users.datalake.ops'
        """
        coroutine = self._build_for_get_index_schema(kind=kind, data_partition_id=data_partition_id)
        return get_event_loop().run_until_complete(coroutine)

    def query(self, data_partition_id: str, query_request: m.QueryRequest = None) -> m.QueryResponse:
        """
        The API supports full text search on string fields, range queries on date, numeric or string fields, along with geo-spatial search. Required roles: 'users.datalake.viewers' or 'users.datalake.editors' or 'users.datalake.admins' or 'users.datalake.ops'. In addition, users must be a member of data groups to access the data.
        """
        coroutine = self._build_for_query(data_partition_id=data_partition_id, query_request=query_request)
        return get_event_loop().run_until_complete(coroutine)

    def query_with_cursor(
        self, data_partition_id: str, cursor_query_request: m.CursorQueryRequest = None
    ) -> m.CursorQueryResponse:
        """
        The API supports full text search on string fields, range queries on date, numeric or string fields, along with geo-spatial search. Required roles: 'users.datalake.viewers' or 'users.datalake.editors' or 'users.datalake.admins' or 'users.datalake.ops'. In addition, users must be a member of data groups to access the data. It can be used to retrieve large numbers of results (or even all results) from a single search request, in much the same way as you would use a cursor on a traditional database.
        """
        coroutine = self._build_for_query_with_cursor(
            data_partition_id=data_partition_id, cursor_query_request=cursor_query_request
        )
        return get_event_loop().run_until_complete(coroutine)
