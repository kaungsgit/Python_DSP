# simple_device.py

from my_states import *


class SimpleDevice(object):
    """
    A simple state machine that mimics the functionality of a device from a
    high level.
    """

    def __init__(self):
        """ Initialize the components. """

        # Start with a default state.
        self.state = LockedState()

    def on_event(self, event):
        """
        This is the bread and butter of the state machine. Incoming events are
        delegated to the given states which then handle the event. The result is
        then assigned as the new state.
        """

        # The next state will be the result of the on_event function.
        self.state = self.state.on_event(event)


class Robot(object):
    """
    A simple state machine that mimics the functionality of a device from a
    high level.
    """

    def __init__(self):
        """ Initialize the components. """

        # Start with a default state.
        self.state = Stopped()

    def on_event(self, event):
        """
        This is the bread and butter of the state machine. Incoming events are
        delegated to the given states which then handle the event. The result is
        then assigned as the new state.
        """

        # The next state will be the result of the on_event function.
        prev_state = self.state
        self.state = self.state.on_event(event)
        curr_state = self.state
        if curr_state is not prev_state:
            self.state.perform_state_action()


if __name__ == '__main__':
    my_simple_dev = SimpleDevice()

    my_simple_dev.on_event('pin_entered')

    my_simple_dev.on_event('device_locked')

    my_robot = Robot()
    my_robot.on_event('seven')
    my_robot.on_event('go')
    my_robot.on_event('go')

    pass
