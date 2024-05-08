from dataclasses import dataclass, field


def fun():
    return 123


@dataclass
class Test:
    a: int
    b: int = field(default_factory=fun)


print(Test(10))
print(Test(11, 12))
