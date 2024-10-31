from ortools.linear_solver import pywraplp
import numpy as np

######### MPSolver #########
def MPSolver(variables, constraints, objectiveCoeffs):
    
    # Instantiate a Glop solver, naming it LinearExample.
    solver = pywraplp.Solver.CreateSolver("GLOP")
    if not solver:
        return

    # Variables
    var = []
    for name, lowerBound, upperBound in variables:
        if lowerBound == "inf":lowerBound = solver.infinity()
        if upperBound == "inf":upperBound = solver.infinity()
        var.append(solver.NumVar(lowerBound, upperBound, name))
    print("Number of variables =", solver.NumVariables())

    # Cconstraints
    for coefs, sign, limit in constraints:
        expression = 0
        for variable, coef  in zip(var, coefs):
            expression += variable *  coef
        if sign == "<=":
            solver.Add(expression <= limit)
        elif sign == ">=":
            solver.Add(expression >= limit)
        elif sign == "==":
            solver.Add(expression == limit)
    print("Number of constraints =", solver.NumConstraints())

    # Objective function
    objectiveExpression = 0
    for variable, coef  in zip(var, objectiveCoeffs):
        objectiveExpression += variable *  coef
    solver.Maximize(objectiveExpression)

    # Solve the system.
    print(f"Solving with {solver.SolverVersion()}")
    status = solver.Solve()

    if status == pywraplp.Solver.OPTIMAL:
        print("Solution:")
        print(f"Objective value = {solver.Objective().Value():0.1f}")
        for variable in var:
            print(f"{variable.name()} = {variable.solution_value():0.1f}")
    elif status == pywraplp.Solver.INFEASIBLE:
        print("The problem is infeasible (no feasible region).")
    elif status == pywraplp.Solver.UNBOUNDED:
        print("The problem is unbounded (feasible region is not bounded).")
    else:
        print("The problem does not have an optimal solution.")
        
    print("\nAdvanced usage:")
    print(f"Problem solved in {solver.wall_time():d} milliseconds")
    print(f"Problem solved in {solver.iterations():d} iterations")

# Exo 1
# MPSolver([["x",0,"inf"], ["y",0,"inf"]], [[[8,10],"<=",3400.0],[[2,3],"<=",960.0]], [22,28])
# Exo 3
# MPSolver([["x",0,"inf"], ["y",0,"inf"]], [[[1,0],"<=",5.0],[[1,1],"<=",10.0],[[-1,1],">=",-2.0]], [3,1])
# Exo 5
# MPSolver([["x",0,"inf"], ["y",0,"inf"]], [[[-1,1],"<=",2.0],[[1,2],"<=",8.0],[[1,0],"<=",6.0]], [1,2])
# Exo 6
# MPSolver([["x",0,"inf"], ["y",0,"inf"]], [[[1,1],">=",4.0],[[-1,1],"<=",4.0],[[-1,2],">=",-4.0]], [3,1])
# Exo 7
# MPSolver([["x",0,"inf"], ["y",0,"inf"]], [[[-1,1],">=",4.0],[[-1,2],"<=",-4.0]], [3,1])


######## Step-by-step execution of the simplex method ########
def print_tableau(tableau, iteration):
    print(f"\n--- Tableau at Iteration {iteration} ---")
    print(tableau)

def simplex(constraints_coefs, righthand, objective_coefs):
    # Get the size of the futur tableau
    num_constraints, num_variables = constraints_coefs.shape
    # Create a empty tableau with the correct size
    tableau = np.zeros((num_constraints + 1, num_variables + num_constraints + 1))
    # Fill the constraints coefs and the identity in the tableau
    tableau[:-1, :-1] = np.hstack((constraints_coefs, np.eye(num_constraints)))
    # Fill the last column of the tableau with the right-hand side of the constraints
    tableau[:-1, -1] = righthand
    print(tableau)
    # Fill the last row of the tableau with the negative objective coefs
    tableau[-1, :num_variables] = -objective_coefs
    
    # Initialize the iteration counter
    iteration = 0
    print_tableau(tableau, iteration)
    
    while True:
        # Verify if the last row (i.e the objective coefs) are all positif or nil, if this is the case the optimal solution has been found
        if all(tableau[-1, :-1] >= 0):
            break

        # Identify one of the non-basic variable which will be convert to a basic variable
        entering = np.argmin(tableau[-1, :-1])

        # Identify the leaving variable using the minimum positive ratio
        ratios = tableau[:-1, -1] / tableau[:-1, entering]
        # Ignore negative raios
        ratios[ratios <= 0] = np.inf
        # We take the minimum ratio
        leaving = np.argmin(ratios)

        # Verify if the problem is unbounded
        if ratios[leaving] == np.inf:
            print("Problem is unbounded.")
            return
        
        # Pivoting
        pivot = tableau[leaving, entering]
        # Normalize the pivot row
        tableau[leaving] /= pivot

        # Updating each rows
        for i in range(tableau.shape[0]):
            if i != leaving:
                tableau[i] -= tableau[i, entering] * tableau[leaving]
        
        iteration += 1
        print_tableau(tableau, iteration)
    
    # Extract solution
    solution = np.zeros(num_variables)
    for i in range(num_constraints):
        basic_var_index = np.where(tableau[i, :num_variables] == 1)[0]
        if len(basic_var_index) == 1:
            solution[basic_var_index[0]] = tableau[i, -1]
    
    optimal_value = tableau[-1, -1]
    print("Optimal solution:", solution)
    print("Optimal value:", optimal_value)

# Exo 1
# simplex(np.array([[8, 10],[2, 3]]), np.array([3400, 960]), np.array([22, 28]))
# Exo 3
# simplex(np.array([[1, 0],[1, 1],[1,-1]]), np.array([5, 10, 2]), np.array([3, 1]))
# Exo 4
# simplex(np.array([[-3,-10],[0,-1],[1,0]]), np.array([-6, -3, 4]), np.array([1, 1]))
# Exo 5
# simplex(np.array([[-1, 1],[1, 2],[1, 0]]), np.array([2, 8, 6]), np.array([1, 2]))
# Exo 6
# simplex(np.array([[-1, -1],[-1, 1],[1, -2]]), np.array([-4, 4, 4]), np.array([3, 1]))
# Exo 7
# simplex(np.array([[1, -1],[-1, 2]]), np.array([-4, -4]), np.array([3, 1]))