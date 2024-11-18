import numpy as np
import heapq
import torch
from functools import partial
from comb_modules.utils import get_neighbourhood_func
from collections import namedtuple
from utils import maybe_parallelize

A_StarOutput = namedtuple("A_StarOutput", ["shortest_path", "is_unique", "transitions"])


def manhattan_heuristic(cur_pos, goal_pos):
    """Manhattan distance as heuristic."""
    return abs(cur_pos[0] - goal_pos[0]) + abs(cur_pos[1] - goal_pos[1])


def a_star(matrix, neighbourhood_fn="4-grid", request_transitions=False):
    x_max, y_max = matrix.shape
    start_x, start_y = 0, 0
    goal_x, goal_y = x_max-1, y_max-1

    neighbors_func = partial(get_neighbourhood_func(neighbourhood_fn), x_max=x_max, y_max=y_max)

    # Initialize with high costs everywhere except start
    costs = np.full_like(matrix, 1.0e10)
    costs[start_x][start_y] = matrix[start_x][start_y]
    num_path = np.zeros_like(matrix)
    num_path[start_x][start_y] = 1
    # Stores list of sets [(f_score, (x,y)), ... ]
    priority_queue = [(matrix[start_x][start_y] + manhattan_heuristic((start_x, start_y),(goal_x, goal_y)),
                    (start_x, start_y))]
    certain = set()
    transitions = dict()
    
    while priority_queue:
        cur_cost, (cur_x, cur_y) = heapq.heappop(priority_queue)

        if (cur_x, cur_y) in certain:
            continue
        
        for x, y in neighbors_func(cur_x, cur_y):
            if (x,y) not in certain:
                # Cost to reach neighbor (x,y)
                new_cost = matrix[x][y] + costs[cur_x][cur_y]
                if new_cost < costs[x][y]:
                    costs[x][y] = new_cost
                    estimated_cost = new_cost + manhattan_heuristic((x,y),  (goal_x, goal_y))
                    heapq.heappush(priority_queue, (estimated_cost, (x,y)))
                    # Store where we we came from
                    transitions[(x,y)] = (cur_x, cur_y)
                elif new_cost == costs[x][y]:
                    num_path[x,y] += 1

        certain.add((cur_x,cur_y))

    # Retrive the path
    cur_x, cur_y = goal_x, goal_y
    on_path = np.zeros_like(matrix)
    on_path[goal_x][goal_y] = 1 
    while (cur_x, cur_y) != (start_x, start_y):
        cur_x, cur_y = transitions[(cur_x, cur_y)]
        on_path[cur_x, cur_y] = 1.0

    is_unique = num_path[goal_x][goal_y] == 1

    if request_transitions:
        return A_StarOutput(shortest_path=on_path, is_unique=is_unique, transitions=transitions)
    else:
        return A_StarOutput(shortest_path=on_path, is_unique=is_unique, transitions=None)


def get_solver(neighbourhood_fn):
    def solver(matrix):
        return a_star(matrix, neighbourhood_fn).shortest_path

    return solver


class ShortestPath(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weights, lambda_val, neighbourhood_fn="8-grid"):
        ctx.lambda_val = lambda_val
        ctx.neighbourhood_fn = neighbourhood_fn
        ctx.solver = get_solver(neighbourhood_fn)
        
        ctx.weights = weights.detach().cpu().numpy()
        ctx.suggested_tours = np.asarray(maybe_parallelize(ctx.solver, arg_list=list(ctx.weights)))
        return torch.from_numpy(ctx.suggested_tours).float().to(weights.device)

    @staticmethod
    def backward(ctx, grad_output):
        assert grad_output.shape == ctx.suggested_tours.shape
        grad_output_numpy = grad_output.detach().cpu().numpy()
        weights_prime = np.maximum(ctx.weights + ctx.lambda_val * grad_output_numpy, 0.0)
        better_paths = np.asarray(maybe_parallelize(ctx.solver, arg_list=list(weights_prime)))
        gradient = -(ctx.suggested_tours - better_paths) / ctx.lambda_val
        return torch.from_numpy(gradient).to(grad_output.device), None, None
