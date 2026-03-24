"""
Jupyter ipywidgets for PEM Electrolyzer PINN demonstrator.

Provides interactive widgets for model selection, hyperparameter tuning,
and execution mode control in Jupyter notebooks.

Follows the teleconnections widget pattern with helper functions,
tuple-based build_widgets, and individual-widget get_args_from_widgets.
"""

import ipywidgets as widgets
import argparse


def create_dropdown(options, value, description):
    """
    Helper function to create a Dropdown widget.
    """
    return widgets.Dropdown(options=options, value=value, description=description)


def create_select_multiple(options, value, description):
    """
    Helper function to create a SelectMultiple widget.
    """
    return widgets.SelectMultiple(options=options, value=value, description=description)


def create_text_input(value, description):
    """
    Helper function to create a Text widget.
    """
    return widgets.Text(value=value, description=description)


def create_int_slider(value, min_val, max_val, step, description):
    """
    Helper function to create an IntSlider widget.
    """
    return widgets.IntSlider(value=value, min=min_val, max=max_val, step=step, description=description)


def create_int_input(value, description):
    """
    Helper function to create an IntText widget.
    """
    return widgets.IntText(value=value, description=description)


def create_float_slider(value, min_val, max_val, step, description, readout_format='.2f'):
    """
    Helper function to create a FloatSlider widget.
    """
    return widgets.FloatSlider(
        value=value, min=min_val, max=max_val, step=step,
        description=description, readout_format=readout_format,
    )


def create_checkbox(value, description):
    """
    Helper function to create a Checkbox widget.
    """
    return widgets.Checkbox(value=value, description=description)


def selector_func(multi_select, options, value, description):
    if multi_select:
        return create_select_multiple(options, value, description)
    return create_dropdown(options, value[0], description)


def build_widgets(models, devices, multi_select=False):
    """
    Build all widgets for PEM experiment configuration.

    Args:
        models (list): List of model names for selection.
        devices (list): List of device options for selection.
        multi_select (bool): If True, uses SelectMultiple instead of Dropdown
            for model and device widgets.

    Returns:
        Tuple of widgets: (model_name, mode, epochs, batch_size, learning_rate,
                           seed, alpha, validation_mode, device, data_dir,
                           results_dir)
    """
    model_name_widget = selector_func(
        multi_select, models, ['teacher'], 'Model Name:'
    )
    mode_widget = create_dropdown(
        ['full', 'quick-test', 'teacher-only', 'ablation'],
        'quick-test',
        'Mode:',
    )
    epochs_widget = create_int_slider(100, 5, 500, 5, 'Epochs:')
    batch_size_widget = create_int_slider(256, 32, 1024, 32, 'Batch Size:')
    learning_rate_widget = create_float_slider(
        -2, -5, -1, 0.1, 'Log10(LR):', readout_format='.1f',
    )
    seed_widget = create_int_input(42, 'Seed:')
    alpha_widget = create_float_slider(
        0.1, 0.0, 1.0, 0.05, 'Alpha:',
    )
    validation_mode_widget = create_dropdown(
        ['keep_out', 'random'], 'keep_out', 'Validation:'
    )
    device_widget = selector_func(
        multi_select, devices, [devices[0]], 'Device:'
    )
    data_dir_widget = create_text_input('dataset/', 'Data Dir:')
    results_dir_widget = create_text_input('results/', 'Results Dir:')

    return (
        model_name_widget,
        mode_widget,
        epochs_widget,
        batch_size_widget,
        learning_rate_widget,
        seed_widget,
        alpha_widget,
        validation_mode_widget,
        device_widget,
        data_dir_widget,
        results_dir_widget,
    )


def create_execution_mode_dropdown():
    """
    Creates a dropdown widget for selecting the execution mode.

    Returns:
        A Dropdown widget for selecting the execution mode.
    """
    execution_mode_dropdown = create_dropdown(
        ['Single Run', 'Parallel Run', 'Cluster Run', 'No Run'],
        'No Run',
        'Execution Mode:',
    )

    def handle_dropdown_change(change):
        """
        Handler for dropdown value change. Outputs mode-specific configurations
        for demonstration.
        """
        config_map = {
            'Single Run': 'Single Run Configuration',
            'Parallel Run': 'Parallel Run Configuration',
            'Cluster Run': 'Cluster Run Configuration',
            'No Run': 'Skip runner, go to analysis of existing results',
        }
        custom_variable = config_map.get(change.new, 'No valid option selected')
        print(custom_variable + ' # For demonstration purposes')

    execution_mode_dropdown.observe(handle_dropdown_change, names='value')
    return execution_mode_dropdown


def get_args_from_widgets(
    model_name_widget,
    mode_widget,
    epochs_widget,
    batch_size_widget,
    learning_rate_widget,
    seed_widget,
    alpha_widget,
    validation_mode_widget,
    device_widget,
    data_dir_widget,
    results_dir_widget,
):
    """
    Convert individual widget values to an argparse.Namespace.

    The learning_rate widget is on a log10 scale, so we convert via 10**value.

    Args:
        model_name_widget: Model name dropdown/select widget.
        mode_widget: Training mode dropdown widget.
        epochs_widget: Epochs IntSlider widget.
        batch_size_widget: Batch size IntSlider widget.
        learning_rate_widget: Log10 learning rate FloatSlider widget.
        seed_widget: Random seed IntText widget.
        alpha_widget: Distillation alpha FloatSlider widget.
        validation_mode_widget: Validation mode dropdown widget.
        device_widget: Device dropdown/select widget.
        data_dir_widget: Data directory Text widget.
        results_dir_widget: Results directory Text widget.

    Returns:
        argparse.Namespace with all experiment parameters.
    """
    args = argparse.Namespace(
        model_name=model_name_widget.value,
        mode=mode_widget.value,
        epochs=epochs_widget.value,
        batch_size=batch_size_widget.value,
        lr=10 ** learning_rate_widget.value,
        seed=seed_widget.value,
        alpha=alpha_widget.value,
        validation=validation_mode_widget.value,
        device=device_widget.value,
        data_dir=data_dir_widget.value,
        results_dir=results_dir_widget.value,
    )

    return args


def display_widgets(widgets_tuple, exec_widget=None):
    """
    Display all widgets in a VBox layout.

    Args:
        widgets_tuple: Tuple of widgets returned by build_widgets.
        exec_widget: Optional execution mode widget to append.

    Returns:
        A VBox containing all widgets.
    """
    items = list(widgets_tuple)
    if exec_widget is not None:
        items.append(exec_widget)
    box = widgets.VBox(items)
    return box
