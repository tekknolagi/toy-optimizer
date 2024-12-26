import unittest
from typing import Optional, Any


class Value:
    def find(self):
        raise NotImplementedError("abstract")

    def _set_forwarded(self, value):
        raise NotImplementedError("abstract")


class Constant(Value):
    def __init__(self, value: Any):
        self.value = value

    def __repr__(self):
        return f"Constant({self.value})"

    def find(self):
        return self

    def _set_forwarded(self, value: Value):
        # if we found out that an Operation is
        # equal to a constant, it's a compiler bug
        # to find out that it's equal to another
        # constant
        assert isinstance(value, Constant) and value.value == self.value


class Operation(Value):
    def __init__(self, name: str, args: list[Value]):
        self.name = name
        self.args = args
        self.forwarded = None
        self.info = None

    def __repr__(self):
        return f"Operation({self.name}, {self.args}, {self.forwarded}, {self.info})"

    def find(self) -> Value:
        # returns the "representative" value of
        # self, in the union-find sense
        op = self
        while isinstance(op, Operation):
            # could do path compression here too
            # but not essential
            next = op.forwarded
            if next is None:
                return op
            op = next
        return op

    def arg(self, index):
        # change to above: return the
        # representative of argument 'index'
        return self.args[index].find()

    def make_equal_to(self, value: Value):
        # this is "union" in the union-find sense,
        # but the direction is important! The
        # representative of the union of Operations
        # must be either a Constant or an operation
        # that we know for sure is not optimized
        # away.

        self.find()._set_forwarded(value)

    def _set_forwarded(self, value: Value):
        self.forwarded = value


class OperationTests(unittest.TestCase):
    def test_construct_example(self):
        # first we need something to represent
        # "a" and "b". In our limited view, we don't
        # know where they come from, so we will define
        # them with a pseudo-operation called "getarg"
        # which takes a number n as an argument and
        # returns the n-th input argument. The proper
        # SSA way to do this would be phi-nodes.

        a = Operation("getarg", [Constant(0)])
        b = Operation("getarg", [Constant(1)])
        # var1 = add(b, 17)
        var1 = Operation("add", [b, Constant(17)])
        # var2 = mul(a, var1)
        var2 = Operation("mul", [a, var1])
        # var3 = add(b, 17)
        var3 = Operation("add", [b, Constant(17)])
        # var4 = add(var2, var3)
        var4 = Operation("add", [var2, var3])

        sequence = [a, b, var1, var2, var3, var4]
        # nothing to test really, it shouldn't crash

    def test_union_find(self):
        # construct three operation, and unify them
        # step by step
        bb = Block()
        a1 = bb.dummy(1)
        a2 = bb.dummy(2)
        a3 = bb.dummy(3)

        # at the beginning, every op is its own
        # representative, that means every
        # operation is in a singleton set
        # {a1} {a2} {a3}
        self.assertIs(a1.find(), a1)
        self.assertIs(a2.find(), a2)
        self.assertIs(a3.find(), a3)

        # now we unify a2 and a1, then the sets are
        # {a1, a2} {a3}
        a2.make_equal_to(a1)
        # they both return a1 as the representative
        self.assertIs(a1.find(), a1)
        self.assertIs(a2.find(), a1)
        # a3 is still different
        self.assertIs(a3.find(), a3)

        # now they are all in the same set {a1, a2, a3}
        a3.make_equal_to(a2)
        self.assertIs(a1.find(), a1)
        self.assertIs(a2.find(), a1)
        self.assertIs(a3.find(), a1)

        # now they are still all the same, and we
        # also learned that they are the same as the
        # constant 6
        # the single remaining set then is
        # {6, a1, a2, a3}
        c = Constant(6)
        a2.make_equal_to(c)
        self.assertIs(a1.find(), c)
        self.assertIs(a2.find(), c)
        self.assertIs(a3.find(), c)

        # union with the same constant again is fine
        a2.make_equal_to(c)


class Block(list):
    def opbuilder(opname):
        def wraparg(arg):
            if not isinstance(arg, Value):
                arg = Constant(arg)
            return arg

        def build(self, *args):
            # construct an Operation, wrap the
            # arguments in Constants if necessary
            op = Operation(opname, [wraparg(arg) for arg in args])
            # add it to self, the basic block
            self.append(op)
            return op

        return build

    # a bunch of operations we support
    add = opbuilder("add")
    mul = opbuilder("mul")
    getarg = opbuilder("getarg")
    dummy = opbuilder("dummy")
    lshift = opbuilder("lshift")
    alloc = opbuilder("alloc")
    load = opbuilder("load")
    store = opbuilder("store")
    print = opbuilder("print")


class BlockTests(unittest.TestCase):
    def test_convenience_block_construction(self):
        bb = Block()
        # a again with getarg, the following line
        # defines the Operation instance and
        # immediately adds it to the basic block bb
        a = bb.getarg(0)
        self.assertEqual(len(bb), 1)
        self.assertEqual(bb[0].name, "getarg")

        # it's a Constant
        self.assertEqual(bb[0].args[0].value, 0)

        # b with getarg
        b = bb.getarg(1)
        # var1 = add(b, 17)
        var1 = bb.add(b, 17)
        # var2 = mul(a, var1)
        var2 = bb.mul(a, var1)
        # var3 = add(b, 17)
        var3 = bb.add(b, 17)
        # var4 = add(var2, var3)
        var4 = bb.add(var2, var3)
        self.assertEqual(len(bb), 6)


def bb_to_str(bb: Block, varprefix: str = "var"):
    # the implementation is not too important,
    # look at the test below to see what the
    # result looks like

    def arg_to_str(arg: Value):
        if isinstance(arg, Constant):
            return str(arg.value)
        else:
            # the key must exist, otherwise it's
            # not a valid SSA basic block:
            # the variable must be defined before
            # its first use
            return varnames[arg]

    varnames = {}
    res = []
    for index, op in enumerate(bb):
        # give the operation a name used while
        # printing:
        var = f"{varprefix}{index}"
        varnames[op] = var
        arguments = ", ".join(arg_to_str(op.arg(i)) for i in range(len(op.args)))
        strop = f"{var} = {op.name}({arguments})"
        res.append(strop)
    return "\n".join(res)


class BlockStrTests(unittest.TestCase):
    def test_basicblock_to_str(self):
        bb = Block()
        var0 = bb.getarg(0)
        var1 = bb.add(5, 4)
        var2 = bb.add(var1, var0)

        self.assertEqual(
            bb_to_str(bb),
            """\
var0 = getarg(0)
var1 = add(5, 4)
var2 = add(var1, var0)""",
        )

        # with a different prefix for the invented
        # variable names:
        self.assertEqual(
            bb_to_str(bb, "x"),
            """\
x0 = getarg(0)
x1 = add(5, 4)
x2 = add(x1, x0)""",
        )

        # and our running example:
        bb = Block()
        a = bb.getarg(0)
        b = bb.getarg(1)
        var1 = bb.add(b, 17)
        var2 = bb.mul(a, var1)
        var3 = bb.add(b, 17)
        var4 = bb.add(var2, var3)

        self.assertEqual(
            bb_to_str(bb, "v"),
            """\
v0 = getarg(0)
v1 = getarg(1)
v2 = add(v1, 17)
v3 = mul(v0, v2)
v4 = add(v1, 17)
v5 = add(v3, v4)""",
        )
        # Note the re-numbering of the variables! We
        # don't attach names to Operations at all, so
        # the printing will just number them in
        # sequence, can sometimes be a source of
        # confusion.


def constfold(bb: Block) -> Block:
    opt_bb = Block()

    for op in bb:
        # basic idea: go over the list and do
        # constant folding of add where possible
        if op.name == "add":
            arg0 = op.arg(0)  # uses .find()
            arg1 = op.arg(1)  # uses .find()
            if isinstance(arg0, Constant) and isinstance(arg1, Constant):
                # can constant-fold! that means we
                # learned a new equality, namely
                # that op is equal to a specific
                # constant
                value = arg0.value + arg1.value
                op.make_equal_to(Constant(value))
                # don't need to have the operation
                # in the optimized basic block
                continue
        # otherwise the operation is not
        # constant-foldable and we put into the
        # output list
        opt_bb.append(op)
    return opt_bb


class ConstFoldTests(unittest.TestCase):
    def test_constfold_simple(self):
        bb = Block()
        var0 = bb.getarg(0)
        var1 = bb.add(5, 4)
        var2 = bb.add(var1, var0)

        opt_bb = constfold(bb)
        self.assertEqual(
            bb_to_str(opt_bb, "optvar"),
            """\
optvar0 = getarg(0)
optvar1 = add(9, optvar0)""",
        )

    def test_constfold_two_ops(self):
        # now it works!
        bb = Block()
        var0 = bb.getarg(0)
        var1 = bb.add(5, 4)
        var2 = bb.add(var1, 10)
        var3 = bb.add(var2, var0)
        opt_bb = constfold(bb)

        self.assertEqual(
            bb_to_str(opt_bb, "optvar"),
            """\
optvar0 = getarg(0)
optvar1 = add(19, optvar0)""",
        )


class VirtualObject:
    def __init__(self):
        self.contents: dict[int, Value] = {}

    def store(self, idx, value):
        self.contents[idx] = value

    def load(self, idx):
        return self.contents[idx]


def get_num(op, index=1):
    assert isinstance(op.arg(index), Constant)
    return op.arg(index).value


def materialize(opt_bb, value: Operation) -> None:
    if isinstance(value, Constant):
        return
    assert isinstance(value, Operation)
    info = value.info
    if info is None:
        # already materialized
        return
    assert isinstance(info, VirtualObject)
    assert value.name == "alloc"
    # put the alloc operation back into the trace
    opt_bb.append(value)
    # only materialize once
    value.info = None
    # put the content back
    for idx, val in sorted(info.contents.items()):
        # materialize recursively
        materialize(opt_bb, val)
        # re-create store operation
        opt_bb.store(value, idx, val)


def optimize_alloc_removal(bb):
    opt_bb = Block()
    for op in bb:
        if op.name == "alloc":
            op.info = VirtualObject()
            continue
        if op.name == "load":
            info = op.arg(0).info
            field = get_num(op)
            op.make_equal_to(info.load(field))
            continue
        if op.name == "store":
            info = op.arg(0).info
            if info:  # virtual
                field = get_num(op)
                info.store(field, op.arg(2))
                continue
            else:  # not virtual
                # first materialize the
                # right hand side
                materialize(opt_bb, op.arg(2))
                # then emit the store via
                # the general path below
        opt_bb.append(op)
    return opt_bb


class AllocationRemovalTests(unittest.TestCase):
    def test_remove_unused_allocation(self):
        bb = Block()
        var0 = bb.getarg(0)
        obj = bb.alloc()
        sto = bb.store(obj, 0, var0)
        var1 = bb.load(obj, 0)
        bb.print(var1)
        opt_bb = optimize_alloc_removal(bb)
        # the virtual object looks like this:
        #  obj
        # ┌──────────┐
        # │ 0: var0  │
        # └──────────┘
        self.assertEqual(
            bb_to_str(opt_bb, "optvar"),
            """\
optvar0 = getarg(0)
optvar1 = print(optvar0)""",
        )

    def test_remove_two_allocations(self):
        bb = Block()
        var0 = bb.getarg(0)
        obj0 = bb.alloc()
        sto1 = bb.store(obj0, 0, var0)
        obj1 = bb.alloc()
        sto2 = bb.store(obj1, 0, obj0)
        var1 = bb.load(obj1, 0)
        var2 = bb.load(var1, 0)
        bb.print(var2)
        # the virtual objects look like this:
        #  obj0
        # ┌──────┐
        # │ 0: ╷ │
        # └────┼─┘
        #      │
        #      ▼
        #     obj1
        #   ┌─────────┐
        #   │ 0: var0 │
        #   └─────────┘
        # therefore
        # var1 is the same as obj0
        # var2 is the same as var0
        opt_bb = optimize_alloc_removal(bb)
        self.assertEqual(
            bb_to_str(opt_bb, "optvar"),
            """\
optvar0 = getarg(0)
optvar1 = print(optvar0)""",
        )

    def test_materialize(self):
        bb = Block()
        var0 = bb.getarg(0)
        obj = bb.alloc()
        sto = bb.store(var0, 0, obj)
        opt_bb = optimize_alloc_removal(bb)
        #  obj is virtual, without any fields
        # ┌───────┐
        # │ empty │
        # └───────┘
        # then we store a reference to obj into
        # field 0 of var0. Since var0 is not virtual,
        # obj escapes, so we have to put it back
        # into the optimized basic block
        self.assertEqual(
            bb_to_str(opt_bb, "optvar"),
            """\
optvar0 = getarg(0)
optvar1 = alloc()
optvar2 = store(optvar0, 0, optvar1)""",
        )

    def test_dont_materialize_twice(self):
        # obj is again an empty virtual object,
        # and we store it into var0 *twice*.
        # this should only materialize it once
        bb = Block()
        var0 = bb.getarg(0)
        obj = bb.alloc()
        sto0 = bb.store(var0, 0, obj)
        sto1 = bb.store(var0, 0, obj)
        opt_bb = optimize_alloc_removal(bb)
        self.assertEqual(
            bb_to_str(opt_bb, "optvar"),
            """\
optvar0 = getarg(0)
optvar1 = alloc()
optvar2 = store(optvar0, 0, optvar1)
optvar3 = store(optvar0, 0, optvar1)""",
        )

    def test_materialize_non_virtuals(self):
        # in this example we store a non-virtual var1
        # into another non-virtual var0
        # this should just lead to no optimization at
        # all
        bb = Block()
        var0 = bb.getarg(0)
        var1 = bb.getarg(1)
        sto = bb.store(var0, 0, var1)
        opt_bb = optimize_alloc_removal(bb)
        self.assertEqual(
            bb_to_str(opt_bb, "optvar"),
            """\
optvar0 = getarg(0)
optvar1 = getarg(1)
optvar2 = store(optvar0, 0, optvar1)""",
        )

    def test_materialization_constants(self):
        # in this example we store the constant 17
        # into the non-virtual var0
        # again, this will not be optimized
        bb = Block()
        var0 = bb.getarg(0)
        sto = bb.store(var0, 0, 17)
        opt_bb = optimize_alloc_removal(bb)
        # the previous line fails so far, triggering
        # the assert:
        # assert not isinstance(value, Constant)
        # in materialize
        self.assertEqual(
            bb_to_str(opt_bb, "optvar"),
            """\
optvar0 = getarg(0)
optvar1 = store(optvar0, 0, 17)""",
        )

    def test_materialize_fields(self):
        bb = Block()
        var0 = bb.getarg(0)
        var1 = bb.getarg(1)
        obj = bb.alloc()
        contents0 = bb.store(obj, 0, 8)
        contents1 = bb.store(obj, 1, var1)
        sto = bb.store(var0, 0, obj)

        # the virtual obj looks like this
        #  obj
        # ┌──────┬──────────┐
        # │ 0: 8 │ 1: var1  │
        # └──────┴──────────┘
        # then it needs to be materialized
        # this is the first example where a virtual
        # object that we want to materialize has any
        # content and is not just an empty object
        opt_bb = optimize_alloc_removal(bb)
        self.assertEqual(
            bb_to_str(opt_bb, "optvar"),
            """\
optvar0 = getarg(0)
optvar1 = getarg(1)
optvar2 = alloc()
optvar3 = store(optvar2, 0, 8)
optvar4 = store(optvar2, 1, optvar1)
optvar5 = store(optvar0, 0, optvar2)""",
        )

    def test_materialize_chained_objects(self):
        bb = Block()
        var0 = bb.getarg(0)
        obj0 = bb.alloc()
        obj1 = bb.alloc()
        contents = bb.store(obj0, 0, obj1)
        const = bb.store(obj1, 0, 1337)
        sto = bb.store(var0, 0, obj0)
        #  obj0
        # ┌──────┐
        # │ 0: ╷ │
        # └────┼─┘
        #      │
        #      ▼
        #     obj1
        #   ┌─────────┐
        #   │ 0: 1337 │
        #   └─────────┘
        # now obj0 escapes
        opt_bb = optimize_alloc_removal(bb)
        self.assertEqual(
            bb_to_str(opt_bb, "optvar"),
            """\
optvar0 = getarg(0)
optvar1 = alloc()
optvar2 = alloc()
optvar3 = store(optvar2, 0, 1337)
optvar4 = store(optvar1, 0, optvar2)
optvar5 = store(optvar0, 0, optvar1)""",
        )

    def test_object_graph_cycles(self):
        bb = Block()
        var0 = bb.getarg(0)
        var1 = bb.alloc()
        var2 = bb.store(var1, 0, var1)
        var3 = bb.store(var0, 1, var1)
        #   ┌────────┐
        #   ▼        │
        #  obj0      │
        # ┌──────┐   │
        # │ 0: ╷ │   │
        # └────┼─┘   │
        #      │     │
        #      └─────┘
        # obj0 points to itself, and then it is
        # escaped
        opt_bb = optimize_alloc_removal(bb)
        # the previous line fails with an
        # InfiniteRecursionError
        # materialize calls itself, infinitely

        # what we want is instead this output:
        self.assertEqual(
            bb_to_str(opt_bb, "optvar"),
            """\
optvar0 = getarg(0)
optvar1 = alloc()
optvar2 = store(optvar1, 0, optvar1)
optvar3 = store(optvar0, 1, optvar1)""",
        )


if __name__ == "__main__":
    unittest.main()
