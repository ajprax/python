class UnsubscribeOnExit:
    def __init__(self, notifications, callback):
        self.notifications = notifications
        self.callback = callback

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.notifications.unsubscribe(self.callback)
