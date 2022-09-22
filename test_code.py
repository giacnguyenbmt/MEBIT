class a:
    x = 0

    def __init__(self, number) -> None:
        print(number + 1)

    @classmethod
    def from_text(cls, text):
        instance = cls(int(text))
        instance.x = 10
        return instance

instance = a.from_text('2')
print(instance.x)

instance = a(2)
print(instance.x)