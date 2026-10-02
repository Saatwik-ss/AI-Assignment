from collections import deque

class Action:
    def __init__(self, name, pos_pre, neg_pre, pos_eff, neg_eff):
        self.name = name
        self.pos_pre = pos_pre
        self.neg_pre = neg_pre
        self.pos_eff = pos_eff
        self.neg_eff = neg_eff
    
    def is_applicable(self, state):
        for pre in self.pos_pre:
            if pre not in state:
                return False
        for pre in self.neg_pre:
            if pre in state:
                return False
        return True
    
    def apply(self, state):
        new_state = state.copy()
        for eff in self.neg_eff:
            new_state.discard(eff)
        for eff in self.pos_eff:
            new_state.add(eff)
        return new_state
    
    def __repr__(self):
        return self.name

initial_state = {"At(Robot,A)", "At(Package,A)"}
goal = {"At(Package,C)"}

actions = [
    Action("Move(A,B)", {"At(Robot,A)"}, set(), {"At(Robot,B)"}, {"At(Robot,A)"}),
    Action("Move(B,A)", {"At(Robot,B)"}, set(), {"At(Robot,A)"}, {"At(Robot,B)"}),
    Action("Move(B,C)", {"At(Robot,B)"}, set(), {"At(Robot,C)"}, {"At(Robot,B)"}),
    Action("Move(C,B)", {"At(Robot,C)"}, set(), {"At(Robot,B)"}, {"At(Robot,C)"}),
    Action("PickUp(Package,A)", {"At(Robot,A)", "At(Package,A)"}, set(), 
           {"Holding(Package)"}, {"At(Package,A)"}),
    Action("PickUp(Package,B)", {"At(Robot,B)", "At(Package,B)"}, set(), 
           {"Holding(Package)"}, {"At(Package,B)"}),
    Action("PickUp(Package,C)", {"At(Robot,C)", "At(Package,C)"}, set(), 
           {"Holding(Package)"}, {"At(Package,C)"}),
    Action("Drop(Package,A)", {"At(Robot,A)", "Holding(Package)"}, set(), 
           {"At(Package,A)"}, {"Holding(Package)"}),
    Action("Drop(Package,B)", {"At(Robot,B)", "Holding(Package)"}, set(), 
           {"At(Package,B)"}, {"Holding(Package)"}),
    Action("Drop(Package,C)", {"At(Robot,C)", "Holding(Package)"}, set(), 
           {"At(Package,C)"}, {"Holding(Package)"}),
]

print(f"Initial: {initial_state}")
print(f"Goal: {goal}")
print(f"Actions: {len(actions)}\n")

class SimplePlanner:
    def __init__(self, initial_state, goal, actions):
        self.initial_state = initial_state
        self.goal = goal
        self.actions = actions
        self.states_explored = 0
    
    def goal_satisfied(self, state):
        return self.goal.issubset(state)
    
    def bfs_plan(self):
        self.states_explored = 0
        queue = deque([(self.initial_state, [])])
        visited = {frozenset(self.initial_state)}
        
        while queue:
            current_state, plan = queue.popleft()
            self.states_explored += 1
            
            if self.goal_satisfied(current_state):
                return plan, current_state
            
            for action in self.actions:
                if action.is_applicable(current_state):
                    new_state = action.apply(current_state)
                    state_frozen = frozenset(new_state)
                    
                    if state_frozen not in visited:
                        visited.add(state_frozen)
                        queue.append((new_state, plan + [action]))
        
        return None, None
    
    def print_plan(self, plan, final_state):
        if plan is None:
            print("No plan found")
            return
        
        print(f"Plan ({len(plan)} actions):\n")
        print(f"S0: {self.initial_state}")
        
        current_state = self.initial_state.copy()
        for i, action in enumerate(plan, 1):
            print(f"\nAction: {action}")
            current_state = action.apply(current_state)
            print(f"S{i}: {current_state}")
        
        print(f"\nGoal satisfied: {self.goal_satisfied(current_state)}")
        print(f"States explored: {self.states_explored}")

planner = SimplePlanner(initial_state, goal, actions)
plan, final_state = planner.bfs_plan()
planner.print_plan(plan, final_state)

# Test 1: Solvable
print("\n" + "="*60)
print("Test 1: Solvable Problem")
print("="*60)

p1 = SimplePlanner(initial_state, goal, actions)
plan1, _ = p1.bfs_plan()

if plan1:
    print(f"✓ Plan found: {[str(a) for a in plan1]}")
    current = initial_state.copy()
    for action in plan1:
        if action.is_applicable(current):
            current = action.apply(current)
        else:
            print(f"✗ Action {action} not applicable")
    print(f"Goal satisfied: {p1.goal_satisfied(current)}")
else:
    print("✗ No plan found")

# Test 2: No PickUp actions
print("\n" + "="*60)
print("Test 2: Impossible (No PickUp)")
print("="*60)

actions_no_pickup = [a for a in actions if "PickUp" not in str(a)]
p2 = SimplePlanner(initial_state, goal, actions_no_pickup)
plan2, _ = p2.bfs_plan()

print(f"Plan found: {plan2 is not None}")
print(f"Expected: False (package can't be moved without PickUp)")
if plan2 is None:
    print("✓ Correctly reported no solution")

# Test 3: Extra actions
print("\n" + "="*60)
print("Test 3: Irrelevant Actions")
print("="*60)

actions_extra = actions + [
    Action("Move(A,C)", {"At(Robot,A)"}, set(), {"At(Robot,C)"}, {"At(Robot,A)"}),
]

p3 = SimplePlanner(initial_state, goal, actions_extra)
plan3, final3 = p3.bfs_plan()

if plan3:
    print(f"✓ Plan found with extra action")
    print(f"Package at C: {'At(Package,C)' in final3}")

# Logic vs Search
print("\n" + "="*60)
print("Logic vs Search")
print("="*60)

state = {"At(Robot,A)", "Holding(Package)"}
print(f"State: {state}\n")

for action in actions[:2]:  # Try first 2 actions
    applicable = action.is_applicable(state)
    print(f"{action}: {'applicable' if applicable else 'not applicable'}")
