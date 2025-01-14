from __future__ import annotations

from typing import TYPE_CHECKING, Any

import cudf
import cupy as cp
import pycuda

from queries.common_utils import (
    check_query_result_pd,
    get_table_path,
    on_second_call,
    run_query_generic,
)
from settings import Settings

if TYPE_CHECKING:
    from collections.abc import Callable

    from dask.dataframe.core import DataFrame

settings = Settings()

dask.config.set(scheduler="threads")


def _read_ds(table_name: str) ->
    path = get_table_path(table_name)

    if settings.run.io_type in ("parquet", "skip"):
        return cudf.read_parquet(path)
    elif settings.run.io_type == "csv":
        df = cudf.read_csv(path)
        for c in df.columns:
            if c.endswith("date"):
                df[c] = df[c].astype("date32[day][pyarrow]")
        return df
    elif settings.run.io_type == "feather":
        return cudf.read_feather(path)
    else:
        msg = f"unsupported file type: {settings.run.io_type!r}"
        return ValueError(msg)


@on_second_call
def get_line_item_ds() -> DataFrame:
    return _read_ds("lineitem")


@on_second_call
def get_orders_ds() -> DataFrame:
    return _read_ds("orders")


@on_second_call
def get_customer_ds() -> DataFrame:
    return _read_ds("customer")


@on_second_call
def get_region_ds() -> DataFrame:
    return _read_ds("region")


@on_second_call
def get_nation_ds() -> DataFrame:
    return _read_ds("nation")


@on_second_call
def get_supplier_ds() -> DataFrame:
    return _read_ds("supplier")


@on_second_call
def get_part_ds() -> DataFrame:
    return _read_ds("part")


@on_second_call
def get_part_supp_ds() -> DataFrame:
    return _read_ds("partsupp")


def run_query(query_number: int, query: Callable[..., Any]) -> None:
    run_query_generic(
        query, query_number, "rapids", query_checker=check_query_result_pd
    )
