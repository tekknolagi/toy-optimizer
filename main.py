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
            if info:  # virtual
                field = get_num(op)
                op.make_equal_to(info.load(field))
                continue
            # otherwise not virtual, use the
            # general path below
        if op.name == "store":
            info = op.arg(0).info
            if info:  # virtual
                field = get_num(op)
                info.store(field, op.arg(2))
                continue
            # not virtual; emit the store via the general path below
        # materialize all the arguments of
        # operations that are put into the
        # output basic block
        for arg in op.args:
            materialize(opt_bb, arg.find())
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

    def test_load_non_virtual(self):
        bb = Block()
        var0 = bb.getarg(0)
        var1 = bb.load(var0, 0)
        bb.print(var1)
        # the next line fails in the line
        # op.make_equal_to(info.load(field))
        # because info is None
        opt_bb = optimize_alloc_removal(bb)
        self.assertEqual(
            bb_to_str(opt_bb, "optvar"),
            """\
optvar0 = getarg(0)
optvar1 = load(optvar0, 0)
optvar2 = print(optvar1)""",
        )

    def test_materialize_on_other_ops(self):
        # materialize not just on store
        bb = Block()
        var0 = bb.getarg(0)
        var1 = bb.alloc()
        var2 = bb.print(var1)
        opt_bb = optimize_alloc_removal(bb)
        self.assertEqual(
            bb_to_str(opt_bb, "optvar"),
            """\
optvar0 = getarg(0)
optvar1 = alloc()
optvar2 = print(optvar1)""",
        )
        # again, the resulting basic block is not in
        # valid SSA form

    def test_sink_allocations(self):
        bb = Block()
        var0 = bb.getarg(0)
        var1 = bb.alloc()
        var2 = bb.store(var1, 0, 123)
        var3 = bb.store(var1, 1, 456)
        var4 = bb.load(var1, 0)
        var5 = bb.load(var1, 1)
        var6 = bb.add(var4, var5)
        var7 = bb.store(var1, 0, var6)
        var8 = bb.store(var0, 1, var1)
        opt_bb = optimize_alloc_removal(bb)
        self.assertEqual(
            bb_to_str(opt_bb, "optvar"),
            """\
optvar0 = getarg(0)
optvar1 = add(123, 456)
optvar2 = alloc()
optvar3 = store(optvar2, 0, optvar1)
optvar4 = store(optvar2, 1, 456)
optvar5 = store(optvar0, 1, optvar2)""",
        )


def eq_value(left: Value, right: Value) -> bool:
    if isinstance(left, Constant) and isinstance(right, Constant):
        return left.value == right.value
    return left is right


def optimize_load_store(bb: Block):
    opt_bb = Block()
    # Stores things we know about the heap at... compile-time. This information
    # can come from either stores or loads.
    compile_time_heap: Dict[Tuple[Value, Offset], Value] = {}
    for op in bb:
        if op.name == "store":
            offset = get_num(op, 1)
            store_info = (op.arg(0), offset)
            current_value = compile_time_heap.get(store_info)
            new_value = op.arg(2)
            if current_value is not None and eq_value(current_value, new_value):
                # No sense storing again if the value inside the field is
                # identical to what we are trying to store. We might know this
                # from a previous store or load.
                # Since we are not writing to the heap in this case, the heap
                # is unchanged, so we don't need to invalidate any prior heap
                # knowledge.
                continue
            # Objects can alias, so we have to remove potentially conflicting
            # writes and reads
            heap_copy = {}
            for key, value in compile_time_heap.items():
                if key[1] != offset:
                    heap_copy[key] = value
            compile_time_heap = heap_copy
            compile_time_heap[store_info] = new_value
        elif op.name == "load":
            load_info = (op.arg(0), get_num(op, 1))
            if load_info in compile_time_heap:
                op.make_equal_to(compile_time_heap[load_info])
                continue
            compile_time_heap[load_info] = op
        opt_bb.append(op)
    return opt_bb


class LoadStoreTests(unittest.TestCase):
    def test_load_after_store_removed(self):
        bb = Block()
        var0 = bb.getarg(0)
        bb.store(var0, 0, 5)
        var1 = bb.load(var0, 0)
        var2 = bb.load(var0, 1)
        bb.print(var1)
        bb.print(var2)
        opt_bb = optimize_load_store(bb)
        self.assertEqual(
            bb_to_str(opt_bb),
            """\
var0 = getarg(0)
var1 = store(var0, 0, 5)
var2 = load(var0, 1)
var3 = print(5)
var4 = print(var2)""",
        )

    def test_loads_between_stores_removed(self):
        bb = Block()
        var0 = bb.getarg(0)
        bb.store(var0, 0, 5)
        var1 = bb.load(var0, 0)
        bb.store(var0, 0, 7)
        var2 = bb.load(var0, 0)
        bb.print(var1)
        bb.print(var2)
        opt_bb = optimize_load_store(bb)
        self.assertEqual(
            bb_to_str(opt_bb),
            """\
var0 = getarg(0)
var1 = store(var0, 0, 5)
var2 = store(var0, 0, 7)
var3 = print(5)
var4 = print(7)""",
        )

    def test_two_stores_same_offset(self):
        bb = Block()
        var0 = bb.getarg(0)
        var1 = bb.getarg(1)
        bb.store(var0, 0, 5)
        bb.store(var1, 0, 7)
        load1 = bb.load(var0, 0)
        load2 = bb.load(var1, 0)
        bb.print(load1)
        bb.print(load2)
        opt_bb = optimize_load_store(bb)
        self.assertEqual(
            bb_to_str(opt_bb),
            """\
var0 = getarg(0)
var1 = getarg(1)
var2 = store(var0, 0, 5)
var3 = store(var1, 0, 7)
var4 = load(var0, 0)
var5 = print(var4)
var6 = print(7)""",
        )

    def test_two_stores_different_offset(self):
        bb = Block()
        var0 = bb.getarg(0)
        var1 = bb.getarg(1)
        bb.store(var0, 0, 5)
        bb.store(var1, 1, 7)
        load1 = bb.load(var0, 0)
        load2 = bb.load(var1, 1)
        bb.print(load1)
        bb.print(load2)
        opt_bb = optimize_load_store(bb)
        self.assertEqual(
            bb_to_str(opt_bb),
            """\
var0 = getarg(0)
var1 = getarg(1)
var2 = store(var0, 0, 5)
var3 = store(var1, 1, 7)
var4 = print(5)
var5 = print(7)""",
        )

    def test_two_loads(self):
        bb = Block()
        var0 = bb.getarg(0)
        var1 = bb.load(var0, 0)
        var2 = bb.load(var0, 0)
        bb.print(var1)
        bb.print(var2)
        opt_bb = optimize_load_store(bb)
        self.assertEqual(
            bb_to_str(opt_bb),
            """\
var0 = getarg(0)
var1 = load(var0, 0)
var2 = print(var1)
var3 = print(var1)""",
        )

    def test_load_store_load(self):
        bb = Block()
        arg1 = bb.getarg(0)
        arg2 = bb.getarg(1)
        var1 = bb.load(arg1, 0)
        bb.store(arg2, 0, 123)
        var2 = bb.load(arg1, 0)
        bb.print(var1)
        bb.print(var2)
        opt_bb = optimize_load_store(bb)
        # Cannot optimize :(
        self.assertEqual(bb_to_str(opt_bb), bb_to_str(bb))

    def test_load_then_store(self):
        bb = Block()
        arg1 = bb.getarg(0)
        var1 = bb.load(arg1, 0)
        bb.store(arg1, 0, var1)
        bb.print(var1)
        opt_bb = optimize_load_store(bb)
        self.assertEqual(
            bb_to_str(opt_bb),
            """\
var0 = getarg(0)
var1 = load(var0, 0)
var2 = print(var1)""",
        )

    # TODO(max): Test above with aliasing objects

    def test_load_then_store_then_load(self):
        bb = Block()
        arg1 = bb.getarg(0)
        var1 = bb.load(arg1, 0)
        bb.store(arg1, 0, var1)
        var2 = bb.load(arg1, 0)
        bb.print(var1)
        bb.print(var2)
        opt_bb = optimize_load_store(bb)
        self.assertEqual(
            bb_to_str(opt_bb),
            """\
var0 = getarg(0)
var1 = load(var0, 0)
var2 = print(var1)
var3 = print(var1)""",
        )

    def test_store_after_store(self):
        bb = Block()
        arg1 = bb.getarg(0)
        bb.store(arg1, 0, 5)
        bb.store(arg1, 0, 5)
        opt_bb = optimize_load_store(bb)
        self.assertEqual(
            bb_to_str(opt_bb),
            """\
var0 = getarg(0)
var1 = store(var0, 0, 5)""",
        )

    def test_load_store_aliasing(self):
        bb = Block()
        arg0 = bb.getarg(0)
        arg1 = bb.getarg(1)
        var0 = bb.load(arg0, 0)
        var1 = bb.load(arg1, 0)
        var2 = bb.store(arg0, 0, var0)
        var3 = bb.load(arg0, 0)
        var4 = bb.load(arg1, 0)
        bb.print(var3)
        bb.print(var4)
        # In the non-aliasing case (arg0 is not arg1), then we can remove:
        # * var2, because we are storing the result of a read;
        # * var3, because we know what we just stored in var2;
        # * var4, because we know the store in var2 did not affect arg1 and we
        #   already have a load
        # In the aliasing case (arg0 is arg1), then we can remove:
        # * var1, because we have already loaded off the same object in var0;
        # * var2, because we are storing the result of a read;
        # * var3, because we know what we just stored in var2;
        # * var4, for the same reason as above
        # Because we don't know if they alias or not, we can only remove the
        # intersection of the above two cases: var2, var3, var4.
        opt_bb = optimize_load_store(bb)
        self.assertEqual(
            bb_to_str(opt_bb),
            """\
var0 = getarg(0)
var1 = getarg(1)
var2 = load(var0, 0)
var3 = load(var1, 0)
var4 = print(var2)
var5 = print(var3)""",
        )


def has_side_effects(op: Operation) -> bool:
    return op.name in {"print", "store"}


def delete_dead_code(bb: Block) -> Block:
    # Mark
    mark = {}
    worklist = []
    for op in bb:
        if has_side_effects(op):  # is critical
            mark[op] = True
            worklist.append(op)
    while worklist:
        op = worklist.pop(0)
        for arg in op.args:
            if isinstance(arg, Constant):
                continue
            arg = arg.find()
            if arg not in mark:
                mark[arg] = True
                worklist.append(arg)
    # Sweep
    return Block([op for op in bb if op in mark])


class DeadCodeEliminationTests(unittest.TestCase):
    def test_delete_unused_op(self):
        bb = Block()
        arg0 = bb.getarg(0)
        opt_bb = delete_dead_code(bb)
        self.assertEqual(bb_to_str(opt_bb), "")

    def test_keep_escaped_op(self):
        bb = Block()
        arg0 = bb.getarg(0)
        bb.print(arg0)
        opt_bb = delete_dead_code(bb)
        self.assertEqual(
            bb_to_str(opt_bb),
            """\
var0 = getarg(0)
var1 = print(var0)""",
        )

    @unittest.expectedFailure
    def test_delete_known_store(self):
        bb = Block()
        var0 = bb.alloc()
        var1 = bb.store(var0, 0, 1)
        opt_bb = delete_dead_code(bb)
        self.assertEqual(bb_to_str(opt_bb), "")

    def test_keep_unknown_store(self):
        bb = Block()
        arg0 = bb.getarg(0)
        var1 = bb.store(arg0, 0, 1)
        opt_bb = delete_dead_code(bb)
        self.assertEqual(
            bb_to_str(opt_bb),
            """\
var0 = getarg(0)
var1 = store(var0, 0, 1)""",
        )


def optimize(bb: Block) -> Block:
    opt_bb = constfold(bb)
    opt_bb = optimize_alloc_removal(opt_bb)
    opt_bb = optimize_load_store(opt_bb)
    opt_bb = delete_dead_code(opt_bb)
    return opt_bb


class OptimizeTests(unittest.TestCase):
    def test_delete_known_store(self):
        bb = Block()
        var0 = bb.alloc()
        var1 = bb.store(var0, 0, 1)
        opt_bb = optimize(bb)
        self.assertEqual(bb_to_str(opt_bb), "")

    def test_keep_unknown_store(self):
        bb = Block()
        arg0 = bb.getarg(0)
        var1 = bb.store(arg0, 0, 1)
        opt_bb = optimize(bb)
        self.assertEqual(
            bb_to_str(opt_bb),
            """\
var0 = getarg(0)
var1 = store(var0, 0, 1)""",
        )


if __name__ == "__main__":
    unittest.main()
