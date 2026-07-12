from pyspark.sql import DataFrame
from pyspark.sql import functions as F


class DataQualityError(Exception):
    """Erro lançado quando algum teste de qualidade falha."""
    pass


def run_data_quality_checks(
    df: DataFrame,
    date_column: str
) -> None:
    """
    Executa testes básicos de qualidade em um Spark DataFrame.

    Valida:
    - DataFrame não vazio;
    - existência de colunas;
    - existência da coluna de data;
    - período mínimo e máximo;
    - valores nulos;
    - linhas duplicadas.

    Caso algum teste falhe, uma exceção é lançada
    e a execução do notebook é interrompida.
    """

    failed_checks = []

    print("=" * 70)
    print("DATA QUALITY CHECK")
    print("=" * 70)

    # ============================================================
    # 1. Quantidade de linhas
    # ============================================================

    row_count = df.count()

    if row_count > 0:
        print(
            f"✅ DataFrame não está vazio: "
            f"{row_count:,} linhas"
        )
    else:
        print("❌ DataFrame está vazio")

        failed_checks.append(
            "DataFrame vazio"
        )

    # ============================================================
    # 2. Colunas
    # ============================================================

    columns = df.columns

    if columns:
        print(
            f"✅ {len(columns)} colunas encontradas"
        )

        print(
            f"   Colunas: {columns}"
        )
    else:
        print(
            "❌ Nenhuma coluna encontrada"
        )

        failed_checks.append(
            "DataFrame sem colunas"
        )

    # ============================================================
    # 3. Tipos das colunas
    # ============================================================

    print("\nTipos das colunas:")

    for column_name, data_type in df.dtypes:

        print(
            f"✅ {column_name}: {data_type}"
        )

    # ============================================================
    # 4. Coluna de data
    # ============================================================

    if date_column in columns:

        print(
            f"\n✅ Coluna de data encontrada: "
            f"{date_column}"
        )

        date_result = (
            df
            .agg(
                F.min(date_column)
                .alias("min_date"),

                F.max(date_column)
                .alias("max_date")
            )
            .first()
        )

        min_date = date_result["min_date"]
        max_date = date_result["max_date"]

        if (
            min_date is not None
            and max_date is not None
        ):

            print(
                f"✅ Período válido: "
                f"{min_date} até {max_date}"
            )

        else:

            print(
                "❌ Não foi possível calcular "
                "o período dos dados"
            )

            failed_checks.append(
                "Período de datas inválido"
            )

    else:

        print(
            f"\n❌ Coluna de data "
            f"'{date_column}' não encontrada"
        )

        failed_checks.append(
            f"Coluna ausente: {date_column}"
        )

    # ============================================================
    # 5. Valores nulos
    # ============================================================

    null_expressions = [
        F.sum(
            F.col(column_name)
            .isNull()
            .cast("integer")
        ).alias(column_name)

        for column_name in columns
    ]

    null_counts = (
        df
        .agg(*null_expressions)
        .first()
        .asDict()
    )

    columns_with_nulls = {
        column_name: count

        for column_name, count
        in null_counts.items()

        if count > 0
    }

    if not columns_with_nulls:

        print(
            "\n✅ Nenhum valor nulo encontrado"
        )

    else:

        print(
            "\n❌ Valores nulos encontrados:"
        )

        for column_name, count in (
            columns_with_nulls.items()
        ):

            print(
                f"   {column_name}: "
                f"{count:,}"
            )

        failed_checks.append(
            "Valores nulos encontrados"
        )

    # ============================================================
    # 6. Linhas duplicadas
    # ============================================================

    distinct_row_count = (
        df
        .dropDuplicates()
        .count()
    )

    duplicate_count = (
        row_count
        - distinct_row_count
    )

    if duplicate_count == 0:

        print(
            "\n✅ Nenhuma linha duplicada encontrada"
        )

    else:

        print(
            f"\n❌ {duplicate_count:,} "
            "linhas duplicadas encontradas"
        )

        failed_checks.append(
            "Linhas duplicadas encontradas"
        )

    # ============================================================
    # Resultado final
    # ============================================================

    print("\n" + "=" * 70)

    if failed_checks:

        print("❌ DATA QUALITY REPROVADO")

        print(
            f"Falhas encontradas: "
            f"{failed_checks}"
        )

        print("=" * 70)

        raise DataQualityError(
            "Os testes de Data Quality falharam. "
            f"Falhas: {failed_checks}"
        )

    print(
        "✅ TODOS OS TESTES "
        "DE DATA QUALITY PASSARAM"
    )

    print("=" * 70)