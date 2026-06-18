from typing import Union, AsyncGenerator, Tuple, Optional, List
from uuid import UUID
from abc import ABC, abstractmethod

from fastapi import Response
from odes_storage.models import Record

from .mime_types import MimeType
from .dataframe_validators import DataFrameValidationFunc
from .consistency_checks import DataConsistencyChecks, BulkInfoForConsistency
from .sessions_storage import Session
from .json_orient import JSONOrient
from .bulk_uri import BulkURI
from .model_chunking import GetDataParams, DataframeBasicDescribe


class BulkIO(ABC):
    """abstract class for bulk I/O"""

    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError()

    @abstractmethod
    def close(self):
        raise NotImplementedError()

    @abstractmethod
    async def write_bulk(
        self,
        ctx,
        data: Union[bytes, AsyncGenerator[bytes, None]],
        content_type: MimeType,
        df_validator_func: DataFrameValidationFunc,
        consistency_checks: DataConsistencyChecks,
        record: Record,
    ) -> Tuple[str, BulkInfoForConsistency]:
        """
        Write an entire bulk
        :param ctx: context instance
        :param data: data corresponding to the serialized dataframe
        :param content_type: either JSON or parquet
        :param df_validator_func: dataframe validation function
        :param consistency_checks: consistency check instance
        :param record: metadata as plain `Record` object
        :return: pair BulkId, bulk description
        """
        raise NotImplementedError()

    @abstractmethod
    async def write_chunk(
        self,
        ctx,
        data: Union[bytes, AsyncGenerator[bytes, None]],
        content_type: MimeType,
        df_validator_func: DataFrameValidationFunc,
        record_id: str,
        session_id: UUID,
        reference_curve: Optional[str]
    ) -> DataframeBasicDescribe:  # TODO to change with BulkInfoForConsistency:
        """
        Write a chunk in a given session
        :param ctx: context instance
        :param data: data corresponding to the serialized dataframe
        :param content_type: either JSON or parquet
        :param df_validator_func: dataframe validation function
        :param record_id: record id
        :param session_id: session id as UUID
        :param reference_curve: reference curve if any else `None`
        :return: chunk dataframe description
        """
        raise NotImplementedError()

    @abstractmethod
    async def write_complete_session(
        self,
        ctx,
        consistency_checks: DataConsistencyChecks,
        record: Record,
        session: Session,
        update_from_bulk_uri: Optional[BulkURI],
        reference_curve: Optional[str]
    ) -> str:
        """
        Complete a session, will run consistency rules and update record creating a new version if successful
        :param ctx: context instance
        :param record: metadata, record as plain `Record` instance
        :param session: session object
        :param update_from_bulk_uri: update from a version, if `None` will perform an overwrite
        :param consistency_checks: consistency check instance
        :param reference_curve: reference curve if any
        :return: bulk id
        """
        raise NotImplementedError()

    @abstractmethod
    async def read_data(
        self,
        ctx,
        record_id: str,
        bulk_uri: BulkURI,
        data_param: GetDataParams,
        accept_type: MimeType,
        orient: Optional[JSONOrient],
    ) -> Response:
        """
        Get data from a given record
        :param ctx: context instance
        :param record_id: record id as string
        :param bulk_uri: bulk uri
        :param data_param: read parameters such as offset, limit, filters ...
        :param accept_type: format requested, either JSON or parquet
        :param orient: if JSON, orient value
        :return: Response to forward directly (TODO: this need to be reviewed, response must be constructed in routers)
        """
        raise NotImplementedError()

    @abstractmethod
    async def get_statistics(
        self,
        ctx,
        record_id: str,
        bulk_uri: str,
        curves_selection: List[str],
    ) -> Response:
        """
        Get data from a given record
        :param ctx: context instance
        :param record_id: record id as string
        :param bulk_uri: bulk uri as string
        :param curves_selection list of columns name requested by the user
        :return: Return bulk statistics if exist
        """
        raise NotImplementedError()

    @abstractmethod
    async def post_statistics(
        self,
        ctx,
        record_id: str,
        bulk_uri: str,
        record_version: int,
    ) -> Response:
        """
        Get data from a given record
        :param ctx: context instance
        :param record_id: record id as string
        :param bulk_uri: bulk uri as string
        :param record_version version of given record
        :return: Return bulk statistics if exist
        """
        raise NotImplementedError()
