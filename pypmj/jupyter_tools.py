# -*- coding: utf-8 -*-
"""Defines classes which are used to provide extended output in the juypter
notebook using widgets, e.g. a progress form.

Authors : Carlo Barth

"""

# Imports
# =============================================================================
import logging
import numpy as np
import sys
from threading import Timer
from . import utils

# =============================================================================
def isnotebook():
    """Checks if the coe is currently executed in the ipython/jupyter-notebook.
    Returns false if it is likely the standard python interpreter.
    
    From (line folded!):
        https://stackoverflow.com/questions/15411967/how-can-i-check-if-code
        -is-executed-in-the-ipython-notebook
    
    """
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter

# =============================================================================
class PerpetualTimer(object):
    """Timer with identical syntax as `threading.Timer`, but starting itself
    continously until `cancel` is called.
    
    """
    def __init__(self, t, hFunction):
        self.t = t
        self.was_cancelled = False
        self.hFunction = hFunction
        self.thread = Timer(self.t, self.handle_function)
        self.thread.setDaemon(True)

    def handle_function(self):
        if self.was_cancelled:
            return
        self.hFunction()
        self.thread = Timer(self.t, self.handle_function)
        self.thread.start()

    def start(self):
        self.thread.start()
        self.was_cancelled = False

    def cancel(self):
        self.thread.cancel()
        self.was_cancelled = True


# =============================================================================
class TerminalProgressDisplay(object):
    """Class that displays status information for a `SimulationSet.run`-method
    call. It displays the remaining runtime (if it can be calculated) and the
    progress of simulation solving in percent and using a progress bar. If
    thtqdm` module is installed it will be used automatically. Otherwise it will
    fall back to a plain python implementation.

    """

    def __init__(self, num_sims, prefix='Progress:', suffix='Finished',
                 decimals=1, bar_length=50):
        self.num_sims = num_sims
        self.prefix = prefix
        self.suffix = suffix
        self.decimals = decimals
        self.bar_length = bar_length
        self.i = 0
        self.t_remaining = None
        self._finished = False
        self._check_tqdm()
        self._initialized = False

    def _check_tqdm(self):
        try:
            from tqdm import tqdm
            self._use_tqdm = True
            self._tqdm = tqdm(total=self.num_sims,
                              desc='\tInitializing progress bar',
                              ncols=self.bar_length+30)
        except ImportError:
            self._use_tqdm = False

    def _print_progress_plain_python(self):
        """Prints the current progress to stdout.

        Based on: https://gist.github.com/vladignatyev/06860ec2040cb497f0f3

        """
        formatStr = "{0:." + str(self.decimals) + "f}"
        percents = formatStr.format(100 * (self.i / float(self.num_sims)))
        filledLength = int(round(
            self.bar_length * self.i / float(self.num_sims)))
        bar = '█' * filledLength + '-' * (self.bar_length - filledLength)
        if self.t_remaining is None:
            tstr = ''
        else:
            if self.i == self.num_sims:
                self.t_remaining = 0.
            tstr = ', approx. remaining time: {}'.format(
                utils.tForm(np.round(self.t_remaining)))
        sys.stdout.write('\r{} |{}| {}% {}{}'.
                         format(self.prefix, bar, percents, self.suffix, tstr))
        if self.i == self.num_sims:
            sys.stdout.write('\n')
            self._finished = True
        sys.stdout.flush()

    def _print_progress_tqdm(self, add_to_value):
        self._tqdm.update(add_to_value)
        self._tqdm.refresh()

        if self.i == self.num_sims:
            self._finished = True
            self._tqdm.close()
            del self._tqdm

    def print_progress(self, add_to_value=None):
        """Prints the current progress."""
        if self._finished:
            return
        if self._use_tqdm:
            if not self._initialized:
                self._tqdm.desc = '\tCurrent progress'
            self._print_progress_tqdm(add_to_value)
        else:
            self._print_progress_plain_python()

    def set_pbar_state(self, add_to_value):
        """Adds `add_to_value` to the current state and updates the
        display.
        """
        self.i += add_to_value
        self.print_progress(add_to_value)

    def update_remaining_time(self, seconds):
        """Sets the current remaining time to `seconds` and updates the
        display."""
        if self._use_tqdm:
            return
        self.t_remaining = seconds
        self.print_progress()


# =============================================================================
class JupyterProgressDisplay(object):
    """Class that displays a jupyter notebook widget if possible, holding
    status information for a `SimulationSet.run`-method call. It displays
    the remaining runtime (if it can be calculated) and the progress of
    simulation solving as text and using a progress bar.
    
    """
    def __init__(self, num_sims=None, show=True):
        
        # Check whether we are in
        self._jupyter_mode = isnotebook()
        self.show = show
        if not self.show:
            return
        if num_sims is None:
            return
        self.logger = logging.getLogger('core.' + self.__class__.__name__)
        self.num_sims = num_sims
        self._timer_ready = False
        if self._jupyter_mode:
            try:
                self._set_up()
                return
            except:
                self._jupyter_mode = False
        
        # This is only for the TerminalProgressDisplay, i.e. if not in
        # jupyter notebook
        self._terminal_display = TerminalProgressDisplay(num_sims)
        if self._terminal_display._use_tqdm:
            return
        self.logger.info('Disabling logging for this run to display '+
                         'the terminal progress bar. Install `tqdm` to have '+
                         'a terminal progress bar that allows simultaneous '+
                         'logging.')
        logging.disable(logging.ERROR)

    
    def _set_up(self):
        """Makes the necessary imports and initializes the jupyter notebook
        widgets."""
        
        # Import the jupyter notebook relevant packages
        import ipywidgets as ipyw
        from IPython.display import display
        
        # Init widgets for the progress bar, progress text and remaining time
        # text
        self._progress_bar = ipyw.IntProgress(min=0, max=self.num_sims)
        self._progress_text = ipyw.Text(value='initializing ...', 
                                        disabled=True)
        self._time_label = ipyw.Text(value='unknown', disabled=True)
        
        # Layout of the items in the form
        form_item_layout = ipyw.Layout(
            display='flex',
            flex_flow='row',
            justify_content='space-between'
        )
        
        # Items in the form
        form_items = [ipyw.Box([ipyw.Label(value='Approx. remaining time:'),
                                self._time_label], layout=form_item_layout),
                      ipyw.Box([ipyw.Label(value='Progress:'), 
                                self._progress_text], layout=form_item_layout),
                      ipyw.Box([ipyw.Label(), 
                                self._progress_bar], layout=form_item_layout)]
        
        # Initialize the form box widget
        form = ipyw.Box(form_items, 
                        layout=ipyw.Layout(display='flex',
                                           flex_flow='column',
                                           align_items='stretch',
                                           width='50%',
                                           border='solid 1px',
                                           padding='10px'))
        
        # Display the form in the jupyter notebook
        display(form)
    
    def set_pbar_state(self, add_to_value=None, description=None,
                        bar_style=None):
        """Updates the progress section with the given options."""
        if not self.show:
            return
        if not self._jupyter_mode:
            if add_to_value is not None:
                self._terminal_display.set_pbar_state(add_to_value=add_to_value)
            return
        if add_to_value is not None:
            try:
                self._progress_bar.value += add_to_value
            except:
                pass
        if description is not None:
            try:
                self._progress_text.value = description
            except:
                pass
        else:
            self._progress_text.value = 'Finished {} of {}'.format(
                                    self._progress_bar.value, self.num_sims)
        if bar_style is not None:
            try:
                self._progress_bar.bar_style = bar_style
            except:
                pass
    
    def _set_up_timer(self):
        """Initializes a PerpetualTimer with a time interval of 1 second,
        starts it and binds it to the `_display_remaining_time` function."""
        self._timer = PerpetualTimer(1.0, self._display_remaining_time)
        self._timer.start()
        self._display_remaining_time()
        self._timer_ready = True
    
    def _display_remaining_time(self):
        """Updates the time label with the current remaining runtime."""
        if self._seconds_remaining <= 0.:
            self._timer.cancel()
            self._timer_ready = False
        if self._jupyter_mode:
            self._time_label.value = utils.tForm(
                                              np.round(self._seconds_remaining))
        else:
            self._terminal_display.update_remaining_time(
                                                        self._seconds_remaining)
        self._seconds_remaining -= 1.
    
    def update_remaining_time(self, seconds):
        """Updates the current remaining runtime value (in seconds)."""
        if not self.show:
            return
        self._seconds_remaining = seconds
        if not self._timer_ready:
            self._set_up_timer()
    
    def set_timer_to_zero(self):
        """Sets the time label to 0 and cancels the timer."""
        if not self.show:
            return
        self._seconds_remaining = 0.
        if self._jupyter_mode:
            self._display_remaining_time()
        else:
            if self._terminal_display._use_tqdm:
                return
            logging.disable(logging.NOTSET)
            self.logger.info('...Logging is enabled again. ')


if __name__ == "__main__":
    pass