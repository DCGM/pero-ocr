from uuid import uuid4
from datetime import datetime, timezone

class UuidService:
    def __call__(self):
        return self.generate_uuid()

    @staticmethod
    def generate_uuid():
        return uuid4()


class DateTimeService:
    def __call__(self, **kwargs):
        return self.get_datetime_now(**kwargs)

    @staticmethod
    def get_datetime_now(**kwargs):
        return datetime.now(timezone.utc)
