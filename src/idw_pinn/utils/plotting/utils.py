"""
Shared utility functions for plotting modules.

Handles filename generation, directory management, LaTeX snippet generation,
and session run ID tracking for consistent output organization.

Updated: January 2026
"""
import os
from datetime import datetime
import uuid


# Module-level run ID - generated once per import/session
_SESSION_RUN_ID = None


def _get_session_run_id():
    """Get or create a session-wide unique run ID."""
    global _SESSION_RUN_ID
    if _SESSION_RUN_ID is None:
        timestamp = datetime.now().strftime('%d%m%y_%H%M%S')
        short_uuid = uuid.uuid4().hex[:4]
        _SESSION_RUN_ID = f"run_{timestamp}_{short_uuid}"
    return _SESSION_RUN_ID


def reset_session_run_id():
    """Reset the session run ID (call at start of new training run)."""
    global _SESSION_RUN_ID
    _SESSION_RUN_ID = None


def generate_unique_filename(base_name: str, extension: str = 'png', 
                             diff_coeff: float = None) -> str:
    """
    Generate unique filename with timestamp and UUID.
    
    Format: {base_name}_{timestamp}_{uuid}.{extension}
    Note: D value no longer included in filename.
    """
    timestamp = datetime.now().strftime('%d%m%y_%H%M%S')
    short_uuid = uuid.uuid4().hex[:4]
    return f"{base_name}_{timestamp}_{short_uuid}.{extension}"


def ensure_output_dirs(output_dir: str):
    """
    Ensure output and subfigures directories exist with session run ID.
    
    Returns:
        tuple: (subfig_dir, run_id)
    """
    os.makedirs(output_dir, exist_ok=True)
    run_id = _get_session_run_id()
    subfig_dir = os.path.join(output_dir, 'subfigures', run_id)
    os.makedirs(subfig_dir, exist_ok=True)
    return subfig_dir, run_id


def append_latex_snippet(output_dir: str, subfigure_paths: list, 
                          figure_type: str, caption: str = '', label: str = '',
                          run_id: str = None):
    """
    Append LaTeX snippet for subfigures to latex_snippets.txt.
    
    Args:
        output_dir: Directory containing the output
        subfigure_paths: List of paths to subfigure PDFs
        figure_type: Type of figure (e.g., 'solution_comparison', 'training_diagnostics')
        caption: Figure caption text
        label: LaTeX label for the figure
        run_id: Shared run ID for this session
    """
    latex_file = os.path.join(output_dir, 'latex_snippets.txt')
    
    if run_id is None:
        run_id = _get_session_run_id()
    
    # Extract just filenames (relative to IDW/Figs/<run_id>/ folder in Overleaf)
    filenames = [os.path.basename(p) for p in subfigure_paths]
    
    # Determine number of columns based on figure type
    # - solution_comparison: 4 columns (12 subfigures in 3 rows of 4)
    # - training_diagnostics: 3 columns (6 subfigures in 2 rows of 3)
    if figure_type == 'training_diagnostics':
        n_cols = min(3, len(filenames))
    else:
        n_cols = min(4, len(filenames))
    
    # Generate timestamp for this entry
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Build LaTeX snippet
    tabular_cols = 'c' * n_cols
    snippet = f"""
% =============================================================================
% {figure_type.upper()} - Generated: {timestamp}
% OVERLEAF DIRECTORY: IDW/Figs/{run_id}
% Upload all subfigures from outputs/subfigures/{run_id}/ to this directory.
% =============================================================================
\\begin{{figure}}[htbp]
\\centering
\\setlength{{\\tabcolsep}}{{2pt}}
\\begin{{tabular}}{{{tabular_cols}}}
"""
    
    # Labels based on number of subfigures
    labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)', 
              '(i)', '(j)', '(k)', '(l)']
    
    for row_start in range(0, len(filenames), n_cols):
        row_files = filenames[row_start:row_start + n_cols]
        row_labels = labels[row_start:row_start + len(row_files)]
        
        # Image row
        img_commands = [f'\\includegraphics[width={0.24}\\textwidth]{{IDW/Figs/{run_id}/{f}}}' 
                       for f in row_files]
        snippet += '    ' + ' &\n    '.join(img_commands) + r' \\' + '\n'
        
        # Label row
        label_commands = [f'\\small {lbl}' for lbl in row_labels]
        snippet += '    ' + ' & '.join(label_commands) + r' \\[0.5em]' + '\n'
    
    snippet += f"""\\end{{tabular}}
\\caption{{{caption if caption else f'{figure_type} results'}}}
\\label{{{label if label else f'fig:{figure_type}'}}}
\\end{{figure}}

% Upload these files to IDW/Figs/{run_id}/:
"""
    for p in subfigure_paths:
        snippet += f"%   {os.path.basename(p)}\n"
    
    snippet += "\n"
    
    # Append to file
    with open(latex_file, 'a') as f:
        f.write(snippet)
    
    print(f"LaTeX snippet appended to: {latex_file}")
    print(f">>> Upload subfigures to Overleaf: IDW/Figs/{run_id}/")