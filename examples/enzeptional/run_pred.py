import logging
import csv
import re
from typing import Tuple, List, Optional
import sys

from gt4sd.frameworks.enzeptional import (
    EnzymeOptimizer,
    SequenceMutator,
    SequenceScorer,
    CrossoverGenerator,
    HuggingFaceEmbedder,
    HuggingFaceModelLoader,
    HuggingFaceTokenizerLoader,
    SelectionGenerator,
)
from gt4sd.configuration import GT4SDConfiguration, sync_algorithm_with_s3


# ----------------------------------------------------------------------
# 1. Ambiente / sincronização
# ----------------------------------------------------------------------
def initialize_environment(model: str = "feasibility") -> Tuple[str, Optional[str]]:
    """Synchronize with GT4SD S3 storage and set up the environment."""
    configuration = GT4SDConfiguration.get_instance()
    sync_algorithm_with_s3("proteins/enzeptional/scorers", module="properties")
    scorer = f"{configuration.gt4sd_local_cache_path}/properties/proteins/enzeptional/scorers/{model}/model.pkl"
    if model == "feasibility":
        return scorer, None
    else:
        scaler = f"{configuration.gt4sd_local_cache_path}/properties/proteins/enzeptional/scorers/{model}/scaler.pkl"
        return scorer, scaler


# ----------------------------------------------------------------------
# 2. Leitura de SMILES
# ----------------------------------------------------------------------
def read_smiles_file(smiles_file: str) -> str:
    """Read a SMILES string from a .smi file."""
    with open(smiles_file, 'r') as f:
        smiles = f.readline().strip()
        if not smiles:
            raise ValueError(f"SMILES file {smiles_file} is empty")
        return smiles


# ----------------------------------------------------------------------
# 3. Leitura do FASTA + parsing dos intervalos (MODIFICADA)
# ----------------------------------------------------------------------
def _parse_domains(domains_str: str) -> List[Tuple[int, int]]:
    """
    Converte a string 'domains:[39-48;161-170;212-219]' em
    [(39,48), (161,170), (212,219)].
    """
    intervals = []
    # Remove colchetes e separa por ';'
    parts = domains_str.strip().lstrip('[').rstrip(']').split(';')
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if '-' not in part:
            raise ValueError(f"Invalid interval format: {part}")
        start_str, end_str = part.split('-')
        start, end = int(start_str), int(end_str)
        if start >= end:
            raise ValueError(f"Start >= end in interval {part}")
        intervals.append((start, end))
    return intervals


def read_fasta_sequences(fasta_file: str) -> List[Tuple[str, str, List[Tuple[int, int]]]]:
    """
    Lê sequências FASTA e extrai os intervalos de domínios presentes no cabeçalho.

    Formato esperado do cabeçalho:
        >UNIPROT_ID ... | domains:[39-48;161-170;...]

    Retorna:
        [(uniprot_id, sequence, [(start1,end1), ...]), ...]
        
    Sequências sem domínios serão ignoradas (não incluídas na lista de retorno).
    """
    sequences = []
    current_id = None
    current_seq_lines = []
    domain_regex = re.compile(r'domains:\[([^]]+)\]')

    with open(fasta_file, 'r') as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            if line.startswith('>'):
                # Salva sequência anterior (se houver)
                if current_id and current_seq_lines:
                    seq = ''.join(current_seq_lines)
                    # SÓ ADICIONA SE TIVER INTERVALOS VÁLIDOS
                    if current_intervals:
                        sequences.append((current_id, seq, current_intervals))
                    else:
                        logging.warning(f"Sequence {current_id} skipped: no valid domains found")
                
                # ---- novo cabeçalho ----
                header = line[1:]                     # remove '>'
                parts = header.split(maxsplit=1)
                current_id = parts[0]                 # primeiro token = ID
                
                # procura domains:
                m = domain_regex.search(header)
                current_intervals = []
                if m:
                    try:
                        domains_str = m.group(1)
                        current_intervals = _parse_domains(domains_str)
                    except ValueError as e:
                        logging.warning(f"Invalid domain format for {current_id}: {e}")
                        current_intervals = []
                else:
                    # SEM DOMÍNIOS - manter current_intervals vazio
                    current_intervals = []
                
                current_seq_lines = []
            else:
                current_seq_lines.append(line)

        # última sequência
        if current_id and current_seq_lines:
            seq = ''.join(current_seq_lines)
            # SÓ ADICIONA SE TIVER INTERVALOS VÁLIDOS
            if current_intervals:
                sequences.append((current_id, seq, current_intervals))
            else:
                logging.warning(f"Sequence {current_id} skipped: no valid domains found")

    if not sequences:
        logging.warning(f"No sequences with valid domains found in {fasta_file}")
    else:
        logging.info(f"Loaded {len(sequences)} sequences with valid domains from {fasta_file}")
    
    return sequences


# ----------------------------------------------------------------------
# 4. Configuração do otimizador
# ----------------------------------------------------------------------
def setup_optimizer(
    substrate_smiles: str,
    product_smiles: str,
    sample_sequence: str,
    scorer_path: str,
    scaler_path: Optional[str],
    intervals: List[Tuple[int, int]],
    concat_order: List[str],
    top_k: int,
    batch_size: int,
    use_xgboost_scorer: bool
) -> EnzymeOptimizer:
    """Set up and return the optimizer with all necessary components configured."""
    language_model_path = "facebook/esm2_t33_650M_UR50D"
    tokenizer_path = "facebook/esm2_t33_650M_UR50D"
    chem_model_path = "seyonec/ChemBERTa-zinc-base-v1"
    chem_tokenizer_path = "seyonec/ChemBERTa-zinc-base-v1"

    model_loader = HuggingFaceModelLoader()
    tokenizer_loader = HuggingFaceTokenizerLoader()

    protein_model = HuggingFaceEmbedder(
        model_loader=model_loader,
        tokenizer_loader=tokenizer_loader,
        model_path=language_model_path,
        tokenizer_path=tokenizer_path,
        cache_dir=None,
        device="cuda",
    )

    chem_model = HuggingFaceEmbedder(
        model_loader=model_loader,
        tokenizer_loader=tokenizer_loader,
        model_path=chem_model_path,
        tokenizer_path=chem_tokenizer_path,
        cache_dir=None,
        device="cuda",
    )

    mutation_config = {
        "type": "language-modeling",
        "embedding_model_path": language_model_path,
        "tokenizer_path": tokenizer_path,
        "unmasking_model_path": language_model_path,
    }

    mutator = SequenceMutator(sequence=sample_sequence, mutation_config=mutation_config)
    mutator.set_top_k(top_k)

    scorer = SequenceScorer(
        protein_model=protein_model,
        scorer_filepath=scorer_path,
        use_xgboost=use_xgboost_scorer,
        scaler_filepath=scaler_path,
    )

    selection_generator = SelectionGenerator()
    crossover_generator = CrossoverGenerator()

    optimizer_config = dict(
        sequence=sample_sequence,
        mutator=mutator,
        scorer=scorer,
        intervals=intervals,
        substrate_smiles=substrate_smiles,
        product_smiles=product_smiles,
        chem_model=chem_model,
        selection_generator=selection_generator,
        crossover_generator=crossover_generator,
        concat_order=concat_order,
        batch_size=batch_size,
        selection_ratio=0.25,
        perform_crossover=True,
        crossover_type="single_point",
        pad_intervals=False,
        minimum_interval_length=6,          # <-- ALTERADO PARA 6
        seed=42,
    )
    return EnzymeOptimizer(**optimizer_config)


# ----------------------------------------------------------------------
# 5. Otimização com retry
# ----------------------------------------------------------------------
def optimize_sequences(optimizer: EnzymeOptimizer, max_attempts: int = 10) -> List[dict]:
    """Optimize sequences using the configured optimizer with retry logic."""
    attempt = 1
    while attempt <= max_attempts:
        try:
            logging.info(f"Attempt {attempt}/{max_attempts} for sequence optimization")
            return optimizer.optimize(
                num_iterations=3, num_sequences=5, num_mutations=5, time_budget=3600
            )
        except Exception as e:
            logging.error(f"Optimization failed on attempt {attempt}: {str(e)}")
            attempt += 1
            if attempt > max_attempts:
                logging.error(f"All {max_attempts} attempts failed for sequence optimization")
                return []
    return []


# ----------------------------------------------------------------------
# 6. Escrita CSV
# ----------------------------------------------------------------------
def write_csv_results(results: List[dict], output_file: str, columns: List[str], append: bool = True):
    """Write results to a CSV file using built-in csv module."""
    mode = 'a' if append else 'w'
    with open(output_file, mode, newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if not append:
            writer.writeheader()
        for result in results:
            writer.writerow(result)


# ----------------------------------------------------------------------
# 7. MAIN
# ----------------------------------------------------------------------
def main(
    fasta_file: str,
    substrate_smiles_file: str,
    product_smiles_file: str,
    output_feasibility: str,
    output_kcat: str
):
    logging.basicConfig(level=logging.INFO)

    # ---------- SMILES ----------
    try:
        substrate_smiles = read_smiles_file(substrate_smiles_file)
        product_smiles = read_smiles_file(product_smiles_file)
        logging.info(f"Loaded SMILES: substrate={substrate_smiles}, product={product_smiles}")
    except ValueError as e:
        logging.error(str(e))
        sys.exit(1)

    # ---------- FASTA + INTERVALS ----------
    try:
        sequences = read_fasta_sequences(fasta_file)
        if not sequences:
            logging.error("No sequences with valid domains found. Exiting.")
            sys.exit(1)
    except Exception as e:
        logging.error(f"Error reading FASTA file: {e}")
        sys.exit(1)

    # ---------- CSV headers ----------
    write_csv_results([], output_feasibility,
                      ["acesso uniprot", "sequence mutant", "Score Feasibility"], append=False)
    write_csv_results([], output_kcat,
                      ["acesso uniprot", "sequence mutant", "Score Kcat"], append=False)
    logging.info(f"Created CSV files: {output_feasibility} and {output_kcat}")

    # ---------- COMMON PARAMS ----------
    top_k = 2
    batch_size = 2
    
    
    # ---------- FEASIBILITY ----------
    
    RUN_FEASIBILITY = False
    
    if RUN_FEASIBILITY:
        concat_order_feas = ["substrate", "sequence", "product"]
        use_xgboost_feas = False
        scorer_path_feas, _ = initialize_environment("feasibility")
        
        for uniprot_id, sample_sequence, intervals in sequences:
            try:
                # VERIFICAÇÃO ADICIONAL: garantir que intervals não está vazio
                
                if not intervals:
                    logging.warning(f"Skipping {uniprot_id}: no domains/intervals")
                    continue
		        
                optimizer = setup_optimizer(
                    substrate_smiles,
                    product_smiles,
                    sample_sequence,
                    scorer_path_feas,
                    None,
                    intervals,
                    concat_order_feas,
                    top_k,
                    batch_size,
                    use_xgboost_feas
                )
                optimized = optimize_sequences(optimizer, max_attempts=10)
                if optimized:
                    results = [
                        {
                            "acesso uniprot": uniprot_id,
                            "sequence mutant": r["sequence"],
                            "Score Feasibility": r["score"]
                        }
                        for r in optimized
                    ]
                    write_csv_results(results, output_feasibility,
                                    ["acesso uniprot", "sequence mutant", "Score Feasibility"], append=True)
                    logging.info(f"Feasibility OK for {uniprot_id}")
                else:
                    logging.warning(f"No feasibility results for {uniprot_id}")
            except Exception as e:
                logging.error(f"Feasibility error for {uniprot_id}: {e}")
                continue
    
    RUN_KCAT = True

    if RUN_KCAT:
        # ---------- KCAT ----------
        concat_order_kcat = ["substrate", "sequence"]
        use_xgboost_kcat = True
        scorer_path_kcat, scaler_path_kcat = initialize_environment("kcat")
        
        for uniprot_id, sample_sequence, intervals in sequences:
            try:
                # VERIFICAÇÃO ADICIONAL: garantir que intervals não está vazio
                if not intervals:
                    logging.warning(f"Skipping {uniprot_id}: no domains/intervals")
                    continue
                
                optimizer = setup_optimizer(
                    substrate_smiles,
                    product_smiles,
                    sample_sequence,
                    scorer_path_kcat,
                    scaler_path_kcat,
                    intervals,
                    concat_order_kcat,
                    top_k,
                    batch_size,
                    use_xgboost_kcat
                )
            
                optimized = optimize_sequences(optimizer, max_attempts=10)
            
                if optimized:
                    results = [
                        {
                            "acesso uniprot": uniprot_id,
                            "sequence mutant": r["sequence"],
                        "Score Kcat": r["score"]
                        }

                        for r in optimized
                    ]
                    write_csv_results(results, output_kcat,
                                      ["acesso uniprot", "sequence mutant", "Score Kcat"], append=True)
                    logging.info(f"Kcat OK for {uniprot_id}")
                else:
                    logging.warning(f"No Kcat results for {uniprot_id}")
            except Exception as e:
                logging.error(f"Kcat error for {uniprot_id}: {e}")
                continue


if __name__ == "__main__":
    if len(sys.argv) != 6:
        print(
            "Usage: python enzeptional_multi_sequence.py "
            "<fasta_file> <substrate_smiles_file> <product_smiles_file> "
            "<output_feasibility> <output_kcat>"
        )
        sys.exit(1)

    fasta_file = sys.argv[1]
    substrate_smiles_file = sys.argv[2]
    product_smiles_file = sys.argv[3]
    output_feasibility = sys.argv[4]
    output_kcat = sys.argv[5]

    main(fasta_file, substrate_smiles_file, product_smiles_file,
         output_feasibility, output_kcat)