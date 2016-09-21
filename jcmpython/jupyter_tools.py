"""Defines classes which are used to provide extended output in the juypter
notebook using widgets, e.g. a progress form.

Authors : Carlo Barth

"""

# Imports
# =============================================================================
import logging
import numpy as np
from threading import Timer
from . import utils


# =============================================================================
class PerpetualTimer(object):
    """Timer with identical syntax as `threading.Timer`, but starting itself
    continously until `cancel` is called.
    
    """
    
    def __init__(self, t, hFunction):
        self.t = t
        self.hFunction = hFunction
        self.thread = Timer(self.t, self.handle_function)

    def handle_function(self):
        self.hFunction()
        self.thread = Timer(self.t, self.handle_function)
        self.thread.start()

    def start(self):
        self.thread.start()

    def cancel(self):
        self.thread.cancel()


# =============================================================================
class JupyterProgressDisplay(object):
    """Class that displays a jupyter notebook widget if possible, holding
    status information for a `SimulationSet.run`-method call. It displays
    the remaining runtime (if it can be calculated) and the progress of
    simulation solving as text and using a progress bar.
    
    """
    def __init__(self, num_sims=None, show=True):
        self._pbar_ready = False
        if not show:
            return
        if num_sims is None:
            return
        self.logger = logging.getLogger('core.' + self.__class__.__name__)
        self.num_sims = num_sims
        try:
            self._set_up()
            self._pbar_ready = True
            self._timer_ready = False
        except:
            self.logger.warn('Unable to set up the progress display.')
    
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
        if not self._pbar_ready:
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
            return
        self._time_label.value = utils.tForm(np.round(self._seconds_remaining))
        self._seconds_remaining -= 1.
    
    def update_remaining_time(self, seconds):
        """Updates the current remaining runtime value (in seconds)."""
        if not self._pbar_ready:
            return
        self._seconds_remaining = seconds
        if not self._timer_ready:
            self._set_up_timer()
    
    def set_timer_to_zero(self):
        """Sets the time label to 0 and cancels the timer."""
        self._timer.cancel()
        self._seconds_remaining = 0.
        self._time_label.value = utils.tForm(0.)
        del self._timer
        self._timer_ready = False


if __name__ == "__main__":
    pass