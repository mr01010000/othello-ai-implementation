import time


class MyPlayer:
    """Minimax with alpha-beta pruning and decent heuristics by Tsimafei Raro

    Decently potent AI centered around minimax with alpha-beta pruning.
    Has sophisticated heuristics that rely on long-term profit rather than on short-term gain.
    """

    def __init__(self, my_color, opponent_color):
        self.my_color = my_color                        # Self's color
        self.opponent_color = opponent_color            # Opponent's color
        self.EMPTY = -1                                 # Empty cell constant

        self.name = 'TimR'                              # username

        self.minimax_max_depth = 0                      # How large to grow minimax tree. Changes at runtime

        self.start_time = 0                             # Start time, used to limit minimiax growth
        self.time_limit = 0.9                           # Time limit after which we terminate minimax

    def is_on_board(self, r, c):
        """Returns true if (r,c) is on board, false otherwise."""
        return 0 <= r <= 7 and 0 <= c <= 7

    def get_opponent_color(self, self_color):
        """Returns the color value of self_color's opponent."""
        return abs(self_color - 1)

    def get_disk_count(self, self_color, board):
        """Returns the amount of self_color's discs on board."""
        count = 0
        for r in range(8):
            for c in range(8):
                if board[r][c] == self_color:
                    count += 1
        return count

    def check_moves(self, board, self_color, coords, delta):
        """Check whether self_color has any valid moves in direction (delta) of (coords)."""
        found_opponent = False
        for i in range(1, 8):
            dr = coords[0] + i * delta[0]
            dc = coords[1] + i * delta[1]

            if self.is_on_board(dr, dc):
                if board[dr][dc] == self_color:
                    break

                elif board[dr][dc] == self.get_opponent_color(self_color):
                    found_opponent = True

                elif board[dr][dc] == self.EMPTY:
                    if found_opponent:
                        return dr, dc
                    else:
                        break

    def find_possible_moves(self, board, self_color):
        """Returns all moves self_color could make on the given board."""
        possible_moves = []
        delta = [(0,-1), (-1,-1), (-1,0), (-1,1), (0,1), (1,1), (1,0), (1,-1)]

        for r in range(len(board)):
            for c in range(len(board[r])):
                if board[r][c] == self_color:
                    for i in range(0, 8):
                        coords = (r, c)

                        found_move = self.check_moves(board, self_color, coords, delta[i])

                        if found_move is not None and found_move not in possible_moves:
                            possible_moves.append(found_move)
        return possible_moves

    def find_flippable_disks(self, board, self_color, coords, delta):
        """Find discs that need to be flipped to update the board."""
        found_opponent = False
        flip_positions = []
        for i in range(1, 8):
            dr = coords[0] + i * delta[0]
            dc = coords[0] + i * delta[1]

            if self.is_on_board(dr, dc):
                if board[dr][dc] == self.EMPTY:
                    break
                elif board[dr][dc] == self.get_opponent_color(self_color):
                    found_opponent = True
                    flip_positions.append((dr, dc))
                elif board[dr][dc] == self_color:
                    if found_opponent:
                        return flip_positions
                    else:
                        break

    def update_board(self, board, self_color, coords):
        """Returns board updated with self_color's move to coords."""
        delta = [(0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1)]

        updated_board = [row[:] for row in board]
        updated_board[coords[0]][coords[1]] = self_color

        flip_positions = []
        for i in range(0, 8):
            flip_positions = self.find_flippable_disks(board, self_color, coords, delta[i])
            if flip_positions is not None:
                for flip_r, flip_c in flip_positions:
                    updated_board[flip_r][flip_c] = self_color
        return updated_board

    def get_stable_disks(self, board, self_color, corner_coords):
        """Returns a rough approximation of the amount of stable disks self_color has in a particular corner."""
        step_row = 1 if corner_coords[0] == 0 else -1
        step_col = 1 if corner_coords[1] == 0 else -1

        bound_row = abs(corner_coords[0] - 7)
        bound_col = abs(corner_coords[1] - 7)

        cur_row = corner_coords[0]
        cur_col = corner_coords[1]

        stable_disks = 0

        while cur_row != bound_row:
            cur_col = corner_coords[1]
            while cur_col != bound_col:
                if board[cur_row][cur_col] == self_color:
                    stable_disks += 1
                else:
                    break
                cur_col += step_col

            # Move bound_col down so we get a diagonally shaped edge
            # Whatever is out of new bound cannot be stable by definition
            if (cur_col > 0 and step_col == -1) or (cur_col < 7 and step_col == 1):
                bound_col = cur_col - 1

                if bound_col < 0 or bound_col > 7:
                    break
            cur_row += step_row

        return stable_disks

    def evaluate(self, board):
        """Returns a numerical value representing how favorable given board is.

        Used at minimax's terminal nodes to chose the optimal move from those available.
        """

        self_moves = self.find_possible_moves(board, self.my_color)
        opponent_moves = self.find_possible_moves(board, self.opponent_color)

        mobility = 0                                # Mobility captures Self's profit in amount of available moves
        disk_parity = 0                             # Disk parity captures Self's profit in raw disk amount
        corners = 0                                 # Corners captures Self's profit in occupied corners
        corner_proximity = 0                        # Corner proximity captures the risk of giving away a free corner
        stability = 0                               # Stability captures Self's profit in unflippable disks

        # Calculating mobility heuristic
        self_immediate_mobility = len(self_moves)
        opponent_immediate_mobility = len(opponent_moves)

        if self_immediate_mobility + opponent_immediate_mobility != 0:
            mobility = 100 * (self_immediate_mobility - opponent_immediate_mobility) / (self_immediate_mobility + opponent_immediate_mobility)

        # Calculate disk parity heuristic
        self_disks = self.get_disk_count(self.my_color, board)
        opponent_disks = self.get_disk_count(self.opponent_color, board)

        disk_parity = 100 * (self_disks - opponent_disks) / (self_disks + opponent_disks)

        # Calculating corner heuristic
        corners_list = [(0,0), (0,7), (7,0), (7,7)]
        self_corners = 0
        opponent_corners = 0

        for corner in corners_list:
            if board[corner[0]][corner[1]] == self.my_color:
                self_corners += 1
            if board[corner[0]][corner[1]] == self.opponent_color:
                opponent_corners += 1

        if self_corners + opponent_corners != 0:
            corners = 100 * (self_corners - opponent_corners) / (self_corners + opponent_corners)

        # Calculating corner proximity heuristic
        corners_proximity_list = [(0, 1), (1, 0), (1, 1), (0, 6), (1, 6), (1, 7), (6, 0), (6, 1), (7, 1), (6, 6), (7, 6), (6, 7)]
        self_corner_proximity = 0
        opponent_corner_proximity = 0

        for cell in corners_proximity_list:
            if board[cell[0]][cell[1]] == self.my_color:
                self_corner_proximity += 1
            if board[cell[0]][cell[1]] == self.opponent_color:
                opponent_corner_proximity += 1

        if self_corner_proximity + opponent_corner_proximity != 0:
            corner_proximity = 100 * (self_corner_proximity - opponent_corner_proximity) / (self_corner_proximity + opponent_corner_proximity)

        # Calculating stability heuristic
        self_stability = self.get_stable_disks(board, self.my_color, (0, 0)) + \
                         self.get_stable_disks(board, self.my_color, (0, 7)) + \
                         self.get_stable_disks(board, self.my_color, (7, 0)) + \
                         self.get_stable_disks(board, self.my_color, (7, 7))

        opponent_stability = self.get_stable_disks(board, self.opponent_color, (0, 0)) + \
                             self.get_stable_disks(board, self.opponent_color, (0, 7)) + \
                             self.get_stable_disks(board, self.opponent_color, (7, 0)) + \
                             self.get_stable_disks(board, self.opponent_color, (7, 7))

        if self_stability + opponent_stability != 0:
            stability = 100 * (self_stability - opponent_stability) / (self_stability + opponent_stability)

        # Calculating the final value
        disk_total = self.get_disk_count(self.my_color, board) + self.get_disk_count(self.opponent_color, board)

        # In early-game, focus on maximal mobility and stability. Avoid amassing too many disks.
        if disk_total < 15:
            heuristic_value = 30 * corners - \
                              15 * corner_proximity + \
                              30 * mobility + \
                              30 * stability

        # In mid-game, focus on capturing corners and further building stability
        elif disk_total < 45:
            heuristic_value = 30 * corners - \
                              15 * corner_proximity + \
                              20 * mobility + \
                              35 * stability

        # In late-game, focus on getting as many discs as possible
        else:
            heuristic_value = 30 * corners + \
                              15 * mobility + \
                              30 * stability + \
                              35 * disk_parity

        return heuristic_value

    def minimax(self, board, depth, self_color, alpha, beta):
        """Generate and search a game tree of depth self.minimax_max_depth"""

        # Reached terminal node, evaluate and pass up the tree.
        # Terminal nodes are either those at max depth, or the last ones we have time for.
        if depth == self.minimax_max_depth or (time.time() - self.start_time > self.time_limit and depth != 0):
            return self.evaluate(board)

        # Reached transient node, keep searching.
        else:

            possible_moves = self.find_possible_moves(board, self_color)

            if possible_moves:

                # Self makes a move
                if depth % 2 == 0:

                    children_nodes = {}
                    value = -10000000

                    for move in possible_moves:

                        updated_board = self.update_board(board, self_color, move)
                        children_nodes[move] = self.minimax(updated_board, depth + 1, self.get_opponent_color(self_color), alpha, beta)

                        temp_value = max(children_nodes[move], alpha)

                        # Alpha-beta pruning
                        if temp_value > value:
                            value = temp_value
                        if temp_value >= beta:
                            break
                        if temp_value > alpha:
                            alpha = temp_value

                    if depth == 0:
                        # Tree has been searched, return all possible moves with their respective worth
                        return children_nodes

                    # Else, just pass current node's worth up the tree
                    return value

                # Opponent makes a move
                else:

                    children_nodes = {}
                    value = 10000000

                    for move in possible_moves:

                        updated_board = self.update_board(board, self_color, move)
                        children_nodes[move] = self.minimax(updated_board, depth + 1, self.get_opponent_color(self_color), alpha, beta)

                        temp_value = min(children_nodes[move], beta)

                        # Alpha-beta pruning
                        if temp_value < value:
                            value = temp_value
                        if temp_value <= alpha:
                            break
                        if temp_value < beta:
                            beta = temp_value

                    # Else, just pass current node's worth up the tree
                    return value

        # Return something even if all hell freezes over.
        return 0

    def move(self, board):
        """Returns a move that was decided to be the most optimal."""
        self.start_time = time.time()
        disk_total = self.get_disk_count(self.my_color, board) + self.get_disk_count(self.opponent_color, board)

        if disk_total < 15:
            # In early-game, we can allow a deeper minimax search since there's not too many possible moves.
            self.minimax_max_depth = 7

        elif disk_total < 45:
            # In mid-game, minimax tree has the most branches. Therefore, we must give it space to breathe.
            self.minimax_max_depth = 5
        else:
            # In the very end-game, minimax tree has the least branches, so we can allow a full search.
            self.minimax_max_depth = 8

        possible_moves = self.find_possible_moves(board, self.my_color)

        # If there's only one move available, return it
        if len(possible_moves) == 1:
            return possible_moves[0]

        # If we can take a corner, take it and don't consider any other options.
        # This rarely backfires and allows to save a tiny bit of time
        corners = [(0,0), (0,7), (7,0), (7,7)]
        for corner in corners:
            if corner in possible_moves:
                return corner

        # Grow a minimax tree to find the best available move
        alpha_init = -10000000
        beta_init = 10000000

        available_moves = self.minimax(board, 0, self.my_color, alpha_init, beta_init)
        print(available_moves)
        if available_moves != 0:
            best_value = max(available_moves.values())
            for move in available_moves:
                if available_moves[move] == best_value:
                    return move

        return None
