"""
Study Runner Module for Prompt Framing Study

Orchestrates the full study pipeline with checkpointing support.
"""
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime
import json
import logging
import numpy as np

from .prompt_manager import PromptManager
from .contextualized_embedding import ContextualizedEmbeddingManager
from .choice_collector import ChoiceCollector, extract_choice_vector
from .problem_generator import ProblemGenerator
from .cost_estimator import CostEstimator
from .validation import validate_config, validate_stan_data, validate_claims_file

logger = logging.getLogger(__name__)


class StudyRunner:
    """
    Orchestrate the complete study pipeline with checkpointing support.
    
    Pipeline:
    1. Load configuration
    2. Generate problems (or load existing)
    3. Generate embeddings for all variants
    4. Collect choices for all variants
    5. Prepare Stan data packages
    6. Fit m_0 model for each variant (optional)
    7. Save results and generate visualizations
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        results_dir: Optional[str] = None
    ):
        """
        Initialize the study runner.
        
        Args:
            config_path: Path to study_config.yaml
            config: Direct config dict (alternative to config_path)
            results_dir: Override results directory
        """
        # Load configuration
        if config is not None:
            self.config = config
        elif config_path is not None:
            import yaml
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            # Use default config path
            config_path = Path(__file__).parent / "configs" / "study_config.yaml"
            if config_path.exists():
                import yaml
                with open(config_path, 'r') as f:
                    self.config = yaml.safe_load(f)
            else:
                self.config = self._default_config()
        
        # Validate configuration
        warnings = validate_config(self.config)
        for w in warnings:
            logger.warning(w)
        
        self.base_dir = Path(__file__).parent
        
        # Setup results directory
        if results_dir:
            self.results_dir = Path(results_dir)
        else:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.results_dir = self.base_dir / "results" / f"run_{timestamp}"
        
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Checkpoint file for resumption
        self.checkpoint_file = self.results_dir / ".checkpoint.json"
        self.checkpoint = self._load_checkpoint()
        
        # Setup logging to file
        self._setup_file_logging()
        
        # Initialize components (lazy - only when needed)
        self._prompt_manager = None
        self._embedding_manager = None
        self._choice_collector = None
        self._problem_generator = None
    
    def _default_config(self) -> Dict[str, Any]:
        """Return default configuration."""
        return {
            "num_problems": 100,
            "K": 3,
            "min_alternatives": 2,
            "max_alternatives": 4,
            "temperature": 0.7,
            "num_repetitions": 1,
            "target_dim": 32,
            "embedding_model": "text-embedding-3-small",
            "llm_model": "gpt-4",
            "provider": "openai",
            "fit_models": False,
            "seed": 42
        }
    
    @property
    def prompt_manager(self) -> PromptManager:
        """Lazy initialization of prompt manager."""
        if self._prompt_manager is None:
            config_path = self.config.get("prompt_variants_path")
            if config_path:
                self._prompt_manager = PromptManager(config_path)
            else:
                self._prompt_manager = PromptManager()
        return self._prompt_manager
    
    @property
    def embedding_manager(self) -> ContextualizedEmbeddingManager:
        """Lazy initialization of embedding manager."""
        if self._embedding_manager is None:
            self._embedding_manager = ContextualizedEmbeddingManager(
                embedding_model=self.config.get("embedding_model", "text-embedding-3-small")
            )
        return self._embedding_manager
    
    @property
    def choice_collector(self) -> ChoiceCollector:
        """Lazy initialization of choice collector."""
        if self._choice_collector is None:
            self._choice_collector = ChoiceCollector(
                llm_model=self.config.get("llm_model", "gpt-4"),
                temperature=self.config.get("temperature", 0.7),
                provider=self.config.get("provider", "openai")
            )
        return self._choice_collector
    
    def _setup_file_logging(self):
        """Configure logging to file in results directory."""
        log_file = self.results_dir / "study.log"
        handler = logging.FileHandler(log_file)
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logging.getLogger().addHandler(handler)
        logger.info(f"Logging to {log_file}")
    
    def _load_checkpoint(self) -> Dict[str, Any]:
        """Load checkpoint if exists."""
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
            logger.info(f"Resuming from checkpoint: completed steps = {checkpoint.get('completed_steps', [])}")
            return checkpoint
        return {"completed_steps": []}
    
    def _save_checkpoint(self, step: str, data: Optional[Dict] = None):
        """Save checkpoint after completing a step."""
        self.checkpoint["completed_steps"].append(step)
        if data:
            self.checkpoint[step] = data
        self.checkpoint["last_updated"] = datetime.now().isoformat()
        
        with open(self.checkpoint_file, 'w') as f:
            json.dump(self.checkpoint, f, indent=2, default=str)
        
        logger.debug(f"Checkpoint saved: {step}")
    
    def _step_completed(self, step: str) -> bool:
        """Check if a step was already completed."""
        return step in self.checkpoint.get("completed_steps", [])
    
    def _load_claims(self) -> List[Dict[str, str]]:
        """Load claims from configured file."""
        claims_path = self.config.get("claims_file")
        if claims_path:
            claims_path = Path(claims_path)
            if not claims_path.is_absolute():
                claims_path = self.base_dir / claims_path
        else:
            claims_path = self.base_dir / "data" / "claims.json"
        
        data = validate_claims_file(str(claims_path))
        return data["claims"]
    
    def _create_stan_data(
        self,
        problems: List[Dict],
        embeddings: Dict[str, np.ndarray],
        choices: List[int],
        claim_id_to_idx: Dict[str, int]
    ) -> Dict[str, Any]:
        """
        Create Stan data package for m_0 model.
        
        Args:
            problems: List of decision problems
            embeddings: Reduced embeddings (claim_id -> vector)
            choices: List of 1-indexed choices
            claim_id_to_idx: Mapping from claim ID to row index
            
        Returns:
            Stan data dict
        """
        M = len(problems)
        R = len(claim_id_to_idx)
        
        # Get embedding dimension from first embedding
        first_emb = next(iter(embeddings.values()))
        D = len(first_emb)
        K = self.config.get("K", 3)
        
        # Build w matrix (R x D)
        w = np.zeros((R, D))
        for claim_id, idx in claim_id_to_idx.items():
            if claim_id in embeddings:
                w[idx] = embeddings[claim_id]
            else:
                logger.warning(f"Missing embedding for claim {claim_id}")
        
        # Build I matrix (M x R) - indicator for which claims are in each problem
        I = np.zeros((M, R), dtype=int)
        for m, problem in enumerate(problems):
            for claim_id in problem["claim_ids"]:
                if claim_id in claim_id_to_idx:
                    I[m, claim_id_to_idx[claim_id]] = 1
        
        # Convert choices to alternative indices within problems
        y = []
        for m, problem in enumerate(problems):
            choice_1idx = choices[m]
            y.append(choice_1idx)
        
        stan_data = {
            "M": M,
            "K": K,
            "D": D,
            "R": R,
            "w": w.tolist(),
            "I": I.tolist(),
            "y": y
        }
        
        # Validate
        validate_stan_data(stan_data, model="m_0")
        
        return stan_data
    
    def estimate_cost(self) -> Dict[str, Any]:
        """Estimate study cost without running."""
        claims = self._load_claims()
        estimator = CostEstimator()
        
        estimate = estimator.estimate_total_study_cost(
            num_claims=len(claims),
            num_problems=self.config.get("num_problems", 100),
            num_variants=len(self.prompt_manager.get_all_variants()),
            num_repetitions=self.config.get("num_repetitions", 1),
            avg_alternatives=(
                self.config.get("min_alternatives", 2) + 
                self.config.get("max_alternatives", 4)
            ) / 2,
            embedding_model=self.config.get("embedding_model", "text-embedding-3-small"),
            llm_model=self.config.get("llm_model", "gpt-4"),
            provider=self.config.get("provider", "openai")
        )
        
        return estimate
    
    def run(self, dry_run: bool = False) -> Dict[str, Any]:
        """
        Execute the full study pipeline with checkpointing.
        
        Args:
            dry_run: If True, only estimate costs without executing
            
        Returns:
            Results dict with all outputs and metadata
        """
        results = {
            "config": self.config,
            "timestamp": datetime.now().isoformat(),
            "results_dir": str(self.results_dir)
        }
        
        # Cost estimation
        estimate = self.estimate_cost()
        results["cost_estimate"] = estimate
        logger.info(f"Estimated cost: ${estimate['total_estimated_cost_usd']:.2f}")
        
        if dry_run:
            logger.info("Dry run complete. Use dry_run=False to execute.")
            return results
        
        try:
            # Step 1: Load claims
            logger.info("Step 1: Loading claims...")
            claims = self._load_claims()
            results["num_claims"] = len(claims)
            
            # Build claim ID to index mapping
            claim_id_to_idx = {c["id"]: i for i, c in enumerate(claims)}
            
            # Step 2: Load or generate problems
            problems_file = self.results_dir / "problems.json"
            if not self._step_completed("problems"):
                logger.info("Step 2: Generating decision problems...")
                
                generator = ProblemGenerator(claims=claims)
                problems = generator.generate_problems(
                    num_problems=self.config.get("num_problems", 100),
                    min_alternatives=self.config.get("min_alternatives", 2),
                    max_alternatives=self.config.get("max_alternatives", 4),
                    seed=self.config.get("seed")
                )
                generator.save_problems(problems, str(problems_file))
                self._save_checkpoint("problems", {"num_problems": len(problems)})
            else:
                logger.info("Step 2: Loading problems from checkpoint...")
                problems, _ = ProblemGenerator.load_problems(str(problems_file))
            
            results["num_problems"] = len(problems)
            
            # Step 3: Get prompt variants
            logger.info("Step 3: Loading prompt variants...")
            variants = self.prompt_manager.get_all_variants()
            results["variants"] = [v.name for v in variants]
            results["num_variants"] = len(variants)
            
            # Step 4: Generate embeddings for all variants
            embeddings_file = self.results_dir / "raw_embeddings.npz"
            if not self._step_completed("embeddings"):
                logger.info("Step 4: Generating contextualized embeddings...")
                all_embeddings = self.embedding_manager.embed_all_variants(claims, variants)
                self.embedding_manager.save_embeddings(all_embeddings, str(embeddings_file))
                self._save_checkpoint("embeddings")
            else:
                logger.info("Step 4: Loading embeddings from checkpoint...")
                all_embeddings = ContextualizedEmbeddingManager.load_embeddings(str(embeddings_file))
            
            # Step 5: Apply dimension reduction
            logger.info("Step 5: Reducing embedding dimensions...")
            target_dim = self.config.get("target_dim", 32)
            reduced_embeddings = {}
            reducers = {}
            
            for variant_name, embs in all_embeddings.items():
                reduced_embeddings[variant_name], reducers[variant_name] = \
                    self.embedding_manager.reduce_dimensions(
                        embs,
                        target_dim=target_dim,
                        method="pca"
                    )
            
            # Save reduced embeddings
            self.embedding_manager.save_embeddings(
                reduced_embeddings,
                str(self.results_dir / "reduced_embeddings.npz")
            )
            
            # Step 6: Collect choices for all variants
            choices_file = self.results_dir / "raw_choices.json"
            if not self._step_completed("choices"):
                logger.info("Step 6: Collecting LLM choices...")
                all_choices = self.choice_collector.collect_all_variants(
                    problems,
                    variants,
                    num_repetitions=self.config.get("num_repetitions", 1)
                )
                self.choice_collector.save_choices(all_choices, str(choices_file))
                self._save_checkpoint("choices")
            else:
                logger.info("Step 6: Loading choices from checkpoint...")
                all_choices = ChoiceCollector.load_choices(str(choices_file))
            
            # Step 7: Prepare Stan data packages
            logger.info("Step 7: Preparing Stan data packages...")
            stan_data_packages = {}
            
            for variant in variants:
                # Extract choice vector for this variant
                y = extract_choice_vector(all_choices[variant.name], problems)
                
                # Create Stan data
                stan_data = self._create_stan_data(
                    problems=problems,
                    embeddings=reduced_embeddings[variant.name],
                    choices=y,
                    claim_id_to_idx=claim_id_to_idx
                )
                
                stan_data_packages[variant.name] = stan_data
                
                # Save individual Stan data file
                stan_file = self.results_dir / f"stan_data_{variant.name}.json"
                with open(stan_file, 'w') as f:
                    json.dump(stan_data, f, indent=2)
            
            self._save_checkpoint("stan_data")
            results["stan_data_files"] = [
                str(self.results_dir / f"stan_data_{v.name}.json") 
                for v in variants
            ]
            
            # Step 8: Fit models (optional)
            if self.config.get("fit_models", False):
                logger.info("Step 8: Fitting m_0 models...")
                results["model_fits"] = self._fit_models(stan_data_packages)
                self._save_checkpoint("models")
            
            # Step 9: Save final metadata
            logger.info("Step 9: Saving results...")
            
            with open(self.results_dir / "run_metadata.json", 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"Study complete! Results saved to {self.results_dir}")
            
            # Clear checkpoint on successful completion
            if self.checkpoint_file.exists():
                self.checkpoint_file.unlink()
            
            return results
            
        except Exception as e:
            logger.error(f"Study failed: {e}", exc_info=True)
            raise
    
    def _fit_models(self, stan_data_packages: Dict[str, Dict]) -> Dict[str, Any]:
        """
        Fit m_0 model for each variant.
        
        Args:
            stan_data_packages: Dict mapping variant_name -> stan_data
            
        Returns:
            Dict with model fit results
        """
        try:
            import cmdstanpy
        except ImportError:
            logger.error("cmdstanpy not installed. Skipping model fitting.")
            return {"error": "cmdstanpy not installed"}
        
        # Find the m_0 model
        model_path = Path(__file__).parent.parent.parent / "models" / "m_0.stan"
        if not model_path.exists():
            logger.error(f"Model file not found: {model_path}")
            return {"error": f"Model not found: {model_path}"}
        
        model = cmdstanpy.CmdStanModel(stan_file=str(model_path))
        
        fit_results = {}
        for variant_name, stan_data in stan_data_packages.items():
            logger.info(f"Fitting model for variant: {variant_name}")
            
            try:
                fit = model.sample(
                    data=stan_data,
                    chains=4,
                    parallel_chains=4,
                    iter_warmup=1000,
                    iter_sampling=1000,
                    seed=self.config.get("seed", 42)
                )
                
                # Extract alpha posterior
                alpha_samples = fit.stan_variable("alpha")
                
                fit_results[variant_name] = {
                    "alpha_mean": float(np.mean(alpha_samples)),
                    "alpha_std": float(np.std(alpha_samples)),
                    "alpha_median": float(np.median(alpha_samples)),
                    "alpha_q05": float(np.percentile(alpha_samples, 5)),
                    "alpha_q95": float(np.percentile(alpha_samples, 95)),
                    "diagnostics": {
                        "num_divergences": int(fit.diagnose().count("divergence")),
                    }
                }
                
                # Save samples
                samples_file = self.results_dir / f"samples_{variant_name}.csv"
                fit.save_csvfiles(str(self.results_dir))
                
            except Exception as e:
                logger.error(f"Model fitting failed for {variant_name}: {e}")
                fit_results[variant_name] = {"error": str(e)}
        
        return fit_results
    
    def resume(self) -> Dict[str, Any]:
        """Resume a previously interrupted run."""
        logger.info("Resuming study from checkpoint...")
        return self.run(dry_run=False)
