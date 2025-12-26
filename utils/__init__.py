"""
Utility modules for the SEU sensitivity project.

This module provides shared utilities for model detection, study design selection,
and default parameter configurations.
"""

# Default parameter generation settings used across analysis modules
DEFAULT_PARAM_GENERATION = {
    'alpha_mean': 0.0,
    'alpha_sd': 1.0,
    'beta_sd': 1.0
}


def is_m1_model(study_design_or_data):
    """
    Check if a study design or data dict is for the m_1 model (has risky choice data).
    
    The m_1 model is identified by the presence of risky choice problems (N > 0),
    in addition to the standard uncertain choice problems (M > 0).
    
    Parameters:
        study_design_or_data: Either a StudyDesign/StudyDesignM1 object or a data dict
        
    Returns:
        bool: True if this is an m_1 model (has risky problems), False otherwise
    """
    if isinstance(study_design_or_data, dict):
        # It's a data dictionary
        return 'N' in study_design_or_data and study_design_or_data.get('N', 0) > 0
    else:
        # It's a StudyDesign object
        return (hasattr(study_design_or_data, 'N') and 
                study_design_or_data.N is not None and 
                study_design_or_data.N > 0)


def get_study_design_class(config):
    """
    Determine the appropriate StudyDesign class based on configuration.
    
    Returns StudyDesignM1 if the config specifies both M and N parameters
    (indicating combined risky and uncertain choice), otherwise returns
    the base StudyDesign class.
    
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