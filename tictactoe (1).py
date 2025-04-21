import numpy as np
import pygame
import sys
import time
import random

pygame.init()
WIDTH, HEIGHT = 600, 650
LINE_WIDTH = 10
BOARD_ROWS, BOARD_COLS = 4, 4
CELL_SIZE = WIDTH // BOARD_COLS
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)

class TicTacToe:
    def __init__(self):
        self.board = np.full((4, 4), '-')
        self.current_player = 'X'
        self.window = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Tic-Tac-Toe 4x4")
        self.window.fill(WHITE)
        self.game_over = False
        self.ai_player = 'O'  # AI is 'O' by default
        self.human_player = 'X'  # Human is 'X' by default
        self.ai_first = False  # Human plays first by default
        self.use_alpha_beta = True  # Use alpha-beta pruning by default
        self.nodes_expanded = 0
        self.font = pygame.font.SysFont('Arial', 20)
        self.algorithm_selected = False
        self.player_selected = False
        self.setup_game()
        
    def setup_game(self):
        # First ask about the algorithm
        self.window.fill(WHITE)
        text3 = self.font.render("Press A: Use Minimax without pruning", True, BLACK)
        text4 = self.font.render("Press B: Use Minimax with Alpha-Beta pruning", True, BLACK)
        
        self.window.blit(text3, (WIDTH//2 - 150, HEIGHT//2 - 30))
        self.window.blit(text4, (WIDTH//2 - 150, HEIGHT//2 + 30))
        pygame.display.update()
        
        # Wait for algorithm selection
        while not self.algorithm_selected:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_a:
                        self.use_alpha_beta = False
                        self.algorithm_selected = True
                    elif event.key == pygame.K_b:
                        self.use_alpha_beta = True
                        self.algorithm_selected = True
        
        # Then ask if AI should play first or second
        self.window.fill(WHITE)
        text1 = self.font.render("Press 1: AI plays first (O)", True, BLACK)
        text2 = self.font.render("Press 2: AI plays second (O)", True, BLACK)
        
        self.window.blit(text1, (WIDTH//2 - 150, HEIGHT//2 - 30))
        self.window.blit(text2, (WIDTH//2 - 150, HEIGHT//2 + 30))
        pygame.display.update()
        
        # Wait for player selection
        while not self.player_selected:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_1:
                        self.ai_first = True
                        self.current_player = self.ai_player
                        self.player_selected = True
                    elif event.key == pygame.K_2:
                        self.ai_first = False
                        self.current_player = self.human_player
                        self.player_selected = True
        
        self.window.fill(WHITE)
        self.draw_grid()
        self.display_info()
        
        # If AI is first, make the first move
        if self.ai_first:
            self.ai_move()
    
    def display_info(self):
        info_rect = pygame.Rect(10, HEIGHT - 50, WIDTH - 20, 50)
        pygame.draw.rect(self.window, WHITE, info_rect)
        # Display game info
        if self.use_alpha_beta:
            algo_text = "Minimax with Alpha-Beta"
        else:
            algo_text = "Minimax without pruning"
        info = self.font.render(f"AI: {self.ai_player} | Human: {self.human_player} | Algorithm: {algo_text}", True, RED)
        nodes = self.font.render(f"        Nodes expanded: {self.nodes_expanded}", True, RED)
        self.window.blit(info, (10, HEIGHT - 40))
        self.window.blit(nodes, (10, HEIGHT - 20))
        pygame.display.update(info_rect)
    
    def draw_grid(self):
        for row in range(1, BOARD_ROWS):
            pygame.draw.line(self.window, BLACK, (0, row * CELL_SIZE), (WIDTH, row * CELL_SIZE), LINE_WIDTH)
        for col in range(1, BOARD_COLS):
            pygame.draw.line(self.window, BLACK, (col * CELL_SIZE, 0), (col * CELL_SIZE, HEIGHT), LINE_WIDTH)
        pygame.display.update()

    def draw_move(self, row, col):
        center_x = col * CELL_SIZE + CELL_SIZE // 2
        center_y = row * CELL_SIZE + CELL_SIZE // 2
        if self.board[row, col] == 'X':
            pygame.draw.line(self.window, RED, (center_x - 50, center_y - 50), (center_x + 50, center_y + 50), LINE_WIDTH)
            pygame.draw.line(self.window, RED, (center_x + 50, center_y - 50), (center_x - 50, center_y + 50), LINE_WIDTH)
        else:
            pygame.draw.circle(self.window, BLUE, (center_x, center_y), 50, LINE_WIDTH)
        pygame.display.update()

    def make_move(self, row, col):
        if self.game_over:
            return
            
        if self.board[row, col] == '-':
            self.board[row, col] = self.current_player
            self.draw_move(row, col)
            
            if self.is_winner(self.current_player):
                self.game_over = True
                winner_text = self.font.render(f"{self.current_player} wins!", True, GREEN)
                self.window.blit(winner_text, (WIDTH//2 - 50, 10))
                pygame.display.update()
                return
            
            if self.is_board_full():
                self.game_over = True
                draw_text = self.font.render("Game Draw!", True, GREEN)
                self.window.blit(draw_text, (WIDTH//2 - 50, 10))
                pygame.display.update()
                return
                
            self.current_player = 'O' if self.current_player == 'X' else 'X'
            self.display_info()
            
            # If it's AI's turn, make the AI move
            if self.current_player == self.ai_player and not self.game_over:
                self.ai_move()

    def is_winner(self, player):
        # Check rows
        for row in range(4):
            if all(self.board[row, col] == player for col in range(4)):
                return True
        
        # Check columns
        for col in range(4):
            if all(self.board[row, col] == player for row in range(4)):
                return True
        
        # Check diagonals
        if all(self.board[i, i] == player for i in range(4)) or all(self.board[i, 3 - i] == player for i in range(4)):
            return True
            
        return False
    
    def is_board_full(self):
        return '-' not in self.board
    
    def ai_move(self):
        start_time = time.time()
        self.nodes_expanded = 0
        
        if self.use_alpha_beta:
            _, best_move = self.minimax_alpha_beta(self.board, 3, float('-inf'), float('inf'), True)
        else:
            _, best_move = self.minimax(self.board, 3, True)
        
        end_time = time.time()
        execution_time = end_time - start_time

         
        time_rect = pygame.Rect(10, HEIGHT - 80, WIDTH - 20, 20)  # Clear previous text area
        pygame.draw.rect(self.window, WHITE, time_rect)
        
        time_text = self.font.render(f"      AI thinking time: {execution_time:.3f} seconds", True, RED)
        self.window.blit(time_text, (10, HEIGHT - 80))
        
        if best_move:
            row, col = best_move
            self.make_move(row, col)
        else:
            # If no move found, make a random valid move
            empty_cells = [(i, j) for i in range(4) for j in range(4) if self.board[i, j] == '-']
            if empty_cells:
                row, col = random.choice(empty_cells)
                self.make_move(row, col)
    
    def evaluate(self, board):
        # Simple evaluation function
        if self.is_winner_state(board, self.ai_player):
            return 10
        elif self.is_winner_state(board, self.human_player):
            return -10
        else:
            return 0
    
    def is_winner_state(self, board, player):
        # Check rows
        for row in range(4):
            if all(board[row, col] == player for col in range(4)):
                return True
        
        # Check columns
        for col in range(4):
            if all(board[row, col] == player for row in range(4)):
                return True
        
        # Check diagonals
        if all(board[i, i] == player for i in range(4)) or all(board[i, 3 - i] == player for i in range(4)):
            return True
            
        return False
    
    def is_board_full_state(self, board):
        return '-' not in board
    
    def get_valid_moves(self, board):
        return [(i, j) for i in range(4) for j in range(4) if board[i, j] == '-']
    
    def minimax(self, board, depth, is_maximizing):
        self.nodes_expanded += 1
        
        if self.is_winner_state(board, self.ai_player):
            return 10 - depth, None
        if self.is_winner_state(board, self.human_player):
            return -10 + depth, None
        if self.is_board_full_state(board) or depth == 0:
            return 0, None
        
        valid_moves = self.get_valid_moves(board)
        
        if is_maximizing:
            best_score = float('-inf')
            best_move = None
            
            for move in valid_moves:
                row, col = move
                board_copy = board.copy()
                board_copy[row, col] = self.ai_player
                
                score, _ = self.minimax(board_copy, depth - 1, False)
                
                if score > best_score:
                    best_score = score
                    best_move = move
            
            return best_score, best_move
        else:
            best_score = float('inf')
            best_move = None
            
            for move in valid_moves:
                row, col = move
                board_copy = board.copy()
                board_copy[row, col] = self.human_player
                
                score, _ = self.minimax(board_copy, depth - 1, True)
                
                if score < best_score:
                    best_score = score
                    best_move = move
            
            return best_score, best_move
    
    def minimax_alpha_beta(self, board, depth, alpha, beta, is_maximizing):
        self.nodes_expanded += 1
        
        if self.is_winner_state(board, self.ai_player):
            return 10 - depth, None
        if self.is_winner_state(board, self.human_player):
            return -10 + depth, None
        if self.is_board_full_state(board) or depth == 0:
            return 0, None
        
        valid_moves = self.get_valid_moves(board)
        
        if is_maximizing:
            best_score = float('-inf')
            best_move = None
            
            for move in valid_moves:
                row, col = move
                board_copy = board.copy()
                board_copy[row, col] = self.ai_player
                
                score, _ = self.minimax_alpha_beta(board_copy, depth - 1, alpha, beta, False)
                
                if score > best_score:
                    best_score = score
                    best_move = move
                
                alpha = max(alpha, best_score)
                if beta <= alpha:
                    break
            
            return best_score, best_move
        else:
            best_score = float('inf')
            best_move = None
            
            for move in valid_moves:
                row, col = move
                board_copy = board.copy()
                board_copy[row, col] = self.human_player
                
                score, _ = self.minimax_alpha_beta(board_copy, depth - 1, alpha, beta, True)
                
                if score < best_score:
                    best_score = score
                    best_move = move
                
                beta = min(beta, best_score)
                if beta <= alpha:
                    break
            
            return best_score, best_move

# Running the game
if __name__ == "__main__":
    game = TicTacToe()
    running = True
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            if event.type == pygame.MOUSEBUTTONDOWN and not game.game_over:
                if game.current_player == game.human_player:
                    x, y = event.pos
                    row = y // CELL_SIZE
                    col = x // CELL_SIZE
                    game.make_move(row, col)
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:  # Restart the game
                    game = TicTacToe()
        
        # Small delay to reduce CPU usage
        pygame.time.delay(50)
    
    pygame.quit()