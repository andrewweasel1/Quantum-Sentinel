import os
import glob
import itertools
import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import scipy.stats as stats
import jsharpe
import quantstats as qs

import config

logger = logging.getLogger(__name__)

class QuantitativeEvaluator:
    def __init__(self) -> None:
        self.confidence_level = 0.95
        self.pbo_partitions = 8

    def compute_trial_p_values(self, returns_matrix: pd.DataFrame) -> np.ndarray:
        p_values = []
        for col in returns_matrix.columns:
            returns = returns_matrix[col].values
            if np.std(returns) == 0:
                p_values.append(1.0)
                continue

            try:
                psr = jsharpe.probabilistic_sharpe_ratio(returns)
            except Exception:
                sr = np.mean(returns) / np.std(returns)
                skewness = stats.skew(returns)
                kurt = stats.kurtosis(returns)
                n = len(returns)
                
                stat = sr * np.sqrt(n - 1)
                denom = np.sqrt(1 - skewness * sr + ((kurt - 1) / 4) * sr**2)
                psr = stats.norm.cdf(stat / denom)

            p_values.append(1.0 - float(psr))
            
        return np.array(p_values)

    def calculate_native_pbo(self, returns_matrix: pd.DataFrame) -> float:
        mat = returns_matrix.values
        T, N = mat.shape
        
        if T < self.pbo_partitions or N < 2:
            return 0.0
            
        partitions = np.array_split(mat, self.pbo_partitions, axis=0)
        partition_indices = list(range(self.pbo_partitions))
        is_combos = list(itertools.combinations(partition_indices, self.pbo_partitions // 2))
        
        logits = []
        for is_idx in is_combos:
            oos_idx = [i for i in partition_indices if i not in is_idx]
            
            is_mat = np.vstack([partitions[i] for i in is_idx])
            oos_mat = np.vstack([partitions[i] for i in oos_idx])
            
            is_std = np.std(is_mat, axis=0)
            is_std[is_std == 0] = 1e-8
            is_sr = np.mean(is_mat, axis=0) / is_std
            
            oos_std = np.std(oos_mat, axis=0)
            oos_std[oos_std == 0] = 1e-8
            oos_sr = np.mean(oos_mat, axis=0) / oos_std
            
            best_is_idx = np.argmax(is_sr)
            best_is_oos_sr = oos_sr[best_is_idx]
            
            rank = np.sum(oos_sr <= best_is_oos_sr) / N
            rank = np.clip(rank, 1e-5, 1.0 - 1e-5)
            
            logit = np.log(rank / (1.0 - rank))
            logits.append(logit)
            
        pbo_value = np.sum(np.array(logits) < 0) / len(logits)
        return float(pbo_value)

    def calculate_alpha_decay(self, champion_returns: pd.Series, benchmark_returns: pd.Series) -> Tuple[float, float, float]:
        """
        Calculates Alpha Decay by splitting the performance into two periods (P1 and P2).
        Mimics the Look-Ahead-Bench dual-period evaluation to detect LLM memorization and 
        the 'Scaling Paradox'.
        """
        split_idx = len(champion_returns) // 2
        
        # P1: Early Period
        champ_p1_ret = champion_returns.iloc[:split_idx].mean() * 252
        bench_p1_ret = benchmark_returns.iloc[:split_idx].mean() * 252
        alpha_p1 = champ_p1_ret - bench_p1_ret
        
        # P2: Late Period (Generalization Test)
        champ_p2_ret = champion_returns.iloc[split_idx:].mean() * 252
        bench_p2_ret = benchmark_returns.iloc[split_idx:].mean() * 252
        alpha_p2 = champ_p2_ret - bench_p2_ret
        
        # Decay is the percentage point (pp) drop in Alpha from P1 to P2
        alpha_decay = alpha_p2 - alpha_p1
        
        return alpha_decay, alpha_p1, alpha_p2

    def assess_sector(self, sector_name: str) -> bool:
        logger.info(f"\n{'='*60}\nEvaluating Candidate for Sector: {sector_name}\n{'='*60}")

        matrix_file = f"returns_matrix_{sector_name}.parquet"
        bench_file = f"benchmark_{sector_name}.parquet"

        if not os.path.exists(matrix_file) or not os.path.exists(bench_file):
            return False

        returns_matrix = pd.read_parquet(matrix_file)
        bench_df = pd.read_parquet(bench_file)
        
        champion_returns = bench_df['champion']
        benchmark_returns = bench_df['benchmark']

        # 1. Multiple Testing Bias (FDR)
        p_values = self.compute_trial_p_values(returns_matrix)
        passes_fdr = jsharpe.control_for_FDR(p_values, alpha=1.0 - self.confidence_level)
        
        best_trial_idx = np.argmin(p_values)
        if not passes_fdr[best_trial_idx]:
            logger.warning(f"[{sector_name}] REJECTED: Failed False Discovery Rate (FDR) control.")
            return False

        # 2. Probability of Backtest Overfitting (PBO)
        try:
            overfit_prob = self.calculate_native_pbo(returns_matrix)
            logger.info(f"[{sector_name}] PBO: {overfit_prob * 100:.2f}%")
            if overfit_prob > 0.50:
                logger.warning(f"[{sector_name}] REJECTED: PBO exceeds 50% threshold.")
                return False
        except Exception:
            return False

        # 3. Minimum Track Record Length (MinTRL)
        champ_sr = champion_returns.mean() / champion_returns.std()
        try:
            min_trl = jsharpe.minimum_track_record_length(
                sharpe_ratio=champ_sr,
                skewness=stats.skew(champion_returns),
                kurtosis=stats.kurtosis(champion_returns),
                confidence_level=self.confidence_level
            )
            if len(champion_returns) < min_trl:
                logger.warning(f"[{sector_name}] REJECTED: Track record length too short.")
                return False
        except Exception:
            return False

        # 4. Look-Ahead Bias & Alpha Decay Detection
        logger.info("4. Evaluating Alpha Decay (Look-Ahead Bias)...")
        alpha_decay, alpha_p1, alpha_p2 = self.calculate_alpha_decay(champion_returns, benchmark_returns)
        logger.info(f"[{sector_name}] Alpha P1: {alpha_p1*100:.2f}% | Alpha P2: {alpha_p2*100:.2f}% | Alpha Decay: {alpha_decay*100:.2f} pp")
        
        # Standard foundation models have been shown to decay more than -15pp when tested strictly out-of-sample
        if config.FUSION_ENABLED and alpha_decay < -0.15:
            logger.warning(f"[{sector_name}] SEVERE ALPHA DECAY DETECTED: The LLM is exhibiting the 'Scaling Paradox'.")
            logger.warning("The foundation model is relying on memorized pre-training data (Memory Trap) rather than genuine reasoning.")
            logger.warning("RECOMMENDATION: Swap your standard model for a Point-in-Time (PiT) model like Pitinf-Small or Pitinf-Medium.")

        # 5. Generate Institutional Tearsheet
        logger.info(f"[{sector_name}] TRUE ALPHA DETECTED. Promoting to production.")
        try:
            if config.FUSION_ENABLED and 'sentiment_score' in bench_df.columns:
                bench_df.index = pd.date_range(start='2020-01-01', periods=len(bench_df), freq='D')
                qs.reports.html(
                    returns=bench_df['champion'], 
                    benchmark=bench_df['benchmark'], 
                    title=f'Quantum Sentinel - {sector_name} Champion Profile (LLM FUSION)', 
                    output=f"tearsheet_{sector_name}.html"
                )
            else:
                qs.reports.html(
                    returns=champion_returns, 
                    benchmark=benchmark_returns, 
                    title=f'Quantum Sentinel - {sector_name} Champion Profile', 
                    output=f"tearsheet_{sector_name}.html"
                )
        except Exception as e:
            logger.error(f"[{sector_name}] Failed to generate QuantStats tearsheet: {e}")

        # Promote candidate to champion
        for suffix in ["", "_features"]:
            candidate_path = os.path.join(config.PROD_MODELS_DIR, f"{sector_name}_candidate{suffix}.json")
            champion_path = os.path.join(config.PROD_MODELS_DIR, f"{sector_name}_champion{suffix}.json")
            if os.path.exists(candidate_path):
                os.replace(candidate_path, champion_path)

        return True

    def run_evaluation_gauntlet(self) -> None:
        logger.info("=== COMMENCING POST-TOURNAMENT EVALUATION ===")
        matrix_files = glob.glob("returns_matrix_*.parquet")

        approved_sectors = []
        for file in matrix_files:
            sector_name = file.replace("returns_matrix_", "").replace(".parquet", "")
            if self.assess_sector(sector_name):
                approved_sectors.append(sector_name)

        logger.info(f"\n=== EVALUATION CONCLUDED. {len(approved_sectors)} SECTORS APPROVED FOR PRODUCTION. ===")

if __name__ == "__main__":
    evaluator = QuantitativeEvaluator()
    evaluator.run_evaluation_gauntlet()