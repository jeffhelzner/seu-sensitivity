"""
Utility modules for the SEU sensitivity project.

This module provides shared utilities for model detection, study design selection,
and default parameter configurations.
"""

import os
import re

# Default parameter generation settings used across analysis modules
DEFAULT_PARAM_GENERATION = {
    'alpha_mean': 0.0,
    'alpha_sd': 1.0,
    'beta_sd': 1.0,
    'omega_mean': 0.0,
    'omega_sd': 1.0,
    'kappa_mean': 0.0,
    'kappa_sd': 0.5,
}

# Registry of estimated (free) parameters for each model
# Does not include transformed parameters (e.g. omega in m_3 is derived)
MODEL_PARAMETERS = {
    'm_0': ['alpha', 'beta', 'delta'],
    'm_01': ['alpha', 'beta', 'delta'],
    'm_1': ['alpha', 'beta', 'delta'],
    'm_2': ['alpha', 'omega', 'beta', 'delta'],
    'm_3': ['alpha', 'kappa', 'beta', 'delta'],
}

# Parameters that are scalar (not matrix/vector) — used for recovery/SBC tracking
MODEL_SCALAR_PARAMETERS = {
    'm_0': ['alpha'],
    'm_01': ['alpha'],
    'm_1': ['alpha'],
    'm_2': ['alpha', 'omega'],
    'm_3': ['alpha', 'kappa'],
}

# Prior hyperparameters required by each model's sim file
MODEL_SIM_HYPERPARAMS = {
    'm_0': ['alpha_mean', 'alpha_sd', 'beta_sd'],
    'm_01': ['alpha_mean', 'alpha_sd', 'beta_sd'],
    'm_1': ['alpha_mean', 'alpha_sd', 'beta_sd'],
    'm_2': ['alpha_mean', 'alpha_sd', 'omega_mean', 'omega_sd', 'beta_sd'],
    'm_3': ['alpha_mean', 'alpha_sd', 'kappa_mean', 'kappa_sd', 'beta_sd'],
}

# Transformed parameters to monitor (not free, but useful for diagnostics)
MODEL_TRANSFORMED_MONITORS = {
    'm_0': ['upsilon'],
    'm_01': ['upsilon'],
    'm_1': ['upsilon'],
    'm_2': ['upsilon'],
    'm_3': ['omega', 'upsilon'],
}


def detect_model_name(model_path):
    """
    Extract the model identifier from a Stan model file path.
    
    Parses the filename to determine the model name (m_0, m_1, m_2, m_3, etc.),
    stripping suffixes like _sim, _sbc.
    
    Parameters:
        model_path (str): Path to a Stan model file, e.g. "models/m_2_sim.stan"
        
    Returns:
        str: Model identifier, e.g. "m_2"
    """
    basename = os.path.basename(model_path)
    # Remove .stan extension
    name = basename.replace('.stan', '')
    # Remove _sim, _sbc suffixes
    name = re.sub(r'_(sim|sbc)$', '', name)
    return name


def get_model_parameters(model_name):
    """
    Get the list of estimated parameter names for a given model.
    
    Parameters:
        model_name (str): Model identifier (e.g. "m_0", "m_1", "m_2", "m_3")
        
    Returns:
        list: Parameter names for the model
    """
    return MODEL_PARAMETERS.get(model_name, MODEL_PARAMETERS['m_0'])


def get_model_scalar_parameters(model_name):
    """
    Get the list of scalar (non-matrix/vector) parameter names for a given model.
    
    Parameters:
        model_name (str): Model identifier
        
    Returns:
        list: Scalar parameter names
    """
    return MODEL_SCALAR_PARAMETERS.get(model_name, MODEL_SCALAR_PARAMETERS['m_0'])


def get_model_sim_hyperparams(model_name):
    """
    Get the list of prior hyperparameter names required by a model's sim file.
    
    Parameters:
        model_name (str): Model identifier
        
    Returns:
        list: Hyperparameter names needed for the sim model's data block
    """
    return MODEL_SIM_HYPERPARAMS.get(model_name, MODEL_SIM_HYPERPARAMS['m_0'])


def get_model_transformed_monitors(model_name):
    """
    Get the list of transformed parameters to monitor for a given model.
    
    Parameters:
        model_name (str): Model identifier
        
    Returns:
        list: Transformed parameter names worth monitoring
    """
    return MODEL_TRANSFORMED_MONITORS.get(model_name, MODEL_TRANSFORMED_MONITORS['m_0'])


def has_risky_data(study_design_or_data):
    """
    Check if data includes risky choice problems (used by m_1, m_2, m_3).
    
    Parameters:
        study_design_or_data: A model name string (e.g. 'm_1'), a data dict, 
                              or a StudyDesign/StudyDesignM1 object
        
    Returns:
        bool: True if risky choice data is present
    """
    if isinstance(study_design_or_data, str):
        # Model name string
        return study_design_or_data in ('m_1', 'm_2', 'm_3')
    elif isinstance(study_design_or_data, dict):
        return 'N' in study_design_or_data and study_design_or_data.get('N', 0) > 0
    else:
        return (hasattr(study_design_or_data, 'N') and 
                study_design_or_data.N is not None and 
                study_design_or_data.N > 0)


def is_m1_model(study_design_or_data):
    """
    Check if a study design or data dict is for a model with risky choice data.
    
    This is a backward-compatible alias for has_risky_data(). Models m_1, m_2,
    and m_3 all use the same data structure with risky choice problems.
    
    Parameters:
        study_design_or_data: Either a StudyDesign/StudyDesignM1 object or a data dict
        
    Returns:
        bool: True if this has risky problems (m_1/m_2/m_3), False otherwise (m_0)
    """
    return has_risky_data(study_design_or_data)


def get_study_design_class(config):
    """
    Determine the appropriate StudyDesign class based on configuration.
    
    Returns StudyDesignM1 if the config specifies both M and N parameters
    (indicating combined risky and uncertain choice, used by m_1/m_2/m_3),
    otherwise returns the base StudyDesign class.
    
    Parameters:
        config (dict): Configuration dictionary, typically containing
                      'study_design_config' with M, N, K, D, R, S parameters
    
    Returns:
        class: StudyDesignM1 if N is specified and > 0, otherwise StudyDesign
    """
    study_config = config.get('study_design_config', config)
    if 'N' in study_config and study_config.get('N', 0) > 0:
        from utils.study_design_m1 import StudyDesignM1
        return StudyDesignM1
    else:
        from utils.study_design import StudyDesign
        return StudyDesign


# Expose study design classes for convenience
# from .study_design import StudyDesign
# from .study_design_m1 import StudyDesignM1