#!/bin/bash

# =========================
# CONFIGURAÇÕES GERAIS
# =========================
FASTA="all_mapping_active_sites.fasta"
SCRIPT="run_pred.py"
PYTHON="python"

# Pasta para resultados
OUTDIR="results"
mkdir -p "${OUTDIR}"

# =========================
# FUNÇÃO DE EXECUÇÃO
# =========================
run_prediction () {
    local TAG=$1
    local SUBSTRATE=$2
    local PRODUCT=$3

    local OUT_FEAS="${OUTDIR}/${TAG}_feasibility.csv"
    local OUT_KCAT="${OUTDIR}/${TAG}_kcat.csv"

    echo "===================================================="
    echo "   Executando otimização: ${TAG}"
    echo "   Substrato: ${SUBSTRATE}"
    echo "   Produto:   ${PRODUCT}"
    echo "===================================================="

    ${PYTHON} ${SCRIPT} \
        ${FASTA} \
        ${SUBSTRATE} \
        ${PRODUCT} \
        ${OUT_FEAS} \
        ${OUT_KCAT}

    if [ $? -ne 0 ]; then
        echo "Erro na execução: ${TAG}"
    else
        echo "Finalizado com sucesso: ${TAG}"
        echo "   → ${OUT_FEAS}"
        echo "   → ${OUT_KCAT}"
    fi

    echo
}

# =========================
# EXECUÇÕES
# =========================

# Inulin (exo)
run_prediction \
    "inulin_exo" \
    "inulin_substrato.smi" \
    "produto_exo_inulin.smi"

# Levan (exo)
run_prediction \
    "levan_exo" \
    "substrato-levan.smi" \
    "produto_levan_exo.smi"

# Invertase
run_prediction \
    "invertase" \
    "invertase_substrato.smi" \
    "invertase_produto.smi"

# Inulin → F4+F4
run_prediction \
    "inulin_endo" \
    "inulin_substrato.smi" \
    "inulin_f4_produto.smi"

echo "Todas as predições foram executadas!"