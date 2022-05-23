# my_states.py

from state import State


# Start of our states
class LockedState(State):
    """
    The state which indicates that there are limited device capabilities.
    """

    def on_event(self, event):
        if event == 'pin_entered':
            return UnlockedState()

        return self


class UnlockedState(State):
    """
    The state which indicates that there are no limitations on device
    capabilities.
    """

    def on_event(self, event):
        if event == 'device_locked':
            return LockedState()

        return self


class Stopped(State):
    """
    The state which indicates that there are no limitations on device
    capabilities.
    """

    def on_event(self, event):
        if event == 'seven':
            return active

        return self


class Active(State):
    """
    The state which indicates that there are no limitations on device
    capabilities.
    """

    def on_event(self, event):
        if event == 'go':
            return going_forward
        elif event == 'back':
            return going_backwards

        return self


class GoingForward(State):
    """
    The state which indicates that there are no limitations on device
    capabilities.
    """

    def on_event(self, event):
        if event == 'stop':
            return stopped
        elif event == 'back':
            return going_backwards

        return self


class GoingBackwards(State):
    """
    The state which indicates that there are no limitations on device
    capabilities.
    """

    def on_event(self, event):
        if event == 'stop':
            return stopped
        elif event == 'go':
            return going_forward

        return self


going_forward = GoingForward()
going_backwards = GoingBackwards()
active = Active()
stopped = Stopped()
