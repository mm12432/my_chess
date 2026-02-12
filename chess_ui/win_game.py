"""
This module handles the main game window and UI logic for the Chinese Chess game using Pygame.
It manages the game loop, event handling (mouse clicks), rendering of the board and pieces,
and the sidebar UI.
"""

# pylint: disable=no-member
import sys
import os
import pygame
from pygame.locals import Rect
from my_chess.chess_core import chessboard, chessman

main_dir = os.path.split(os.path.abspath(__file__))[0]
BOARD_WIDTH = 720
BOARD_HEIGHT = 800
SIDEBAR_WIDTH = 280
SCREENRECT = Rect(0, 0, BOARD_WIDTH + SIDEBAR_WIDTH, BOARD_HEIGHT)

# Button Constants
BUTTON_WIDTH = 120
BUTTON_HEIGHT = 45
BUTTON_X = BOARD_WIDTH + (SIDEBAR_WIDTH - BUTTON_WIDTH) // 2
BTN_UNDO_RECT = Rect(BUTTON_X, 650, BUTTON_WIDTH, BUTTON_HEIGHT)
BTN_RESTART_RECT = Rect(BUTTON_X, 710, BUTTON_WIDTH, BUTTON_HEIGHT)
BTN_SAVE_RECT = Rect(BUTTON_X, 590, BUTTON_WIDTH, BUTTON_HEIGHT)


def load_image(file, name=None):
    """
    Loads an image from the disk.
    """
    if name is None:
        # loads an image, prepares it for play
        file = os.path.join(main_dir, "img", file)
    try:
        surface = pygame.image.load(file)
    except pygame.error:
        # Fallback for systems without image support or missing files
        surface = pygame.Surface((80, 80))
        surface.fill((255, 0, 255))
    return surface.convert()


def load_sound(filename):
    """
    Loads a sound file from the disk.
    """
    filename = os.path.join(main_dir, "sounds", filename)
    try:
        sound = pygame.mixer.Sound(filename)
    except pygame.error:

        class MockSound:
            """
            A mock sound class for when the mixer module is not initialized.
            """

            def play(self):
                """Mock play method."""

        sound = MockSound()
    return sound


def load_images(*files):
    """
    Loads multiple images from the disk.
    """
    imgs = []
    for file in files:
        imgs.append(load_image(file))
    return imgs


class ChessmanSprite(pygame.sprite.Sprite):
    """
    Represents a chess piece sprite for Pygame.
    """

    is_selected = False
    images = []
    is_transparent = False

    def __init__(self, images, kill_sound, piece):
        pygame.sprite.Sprite.__init__(self)
        self.chessman = piece
        self.images = images
        self.image = self.images[0]
        self.rect = Rect(piece.col_num * 80, (9 - piece.row_num) * 80, 80, 80)
        self.move_sound = load_sound("move.mp3")
        self.kill_sound = kill_sound

    def move(self, col_num, row_num):
        """
        Moves the sprite to a new position on the board.
        """
        old_col_num = self.chessman.col_num
        old_row_num = self.chessman.row_num
        is_correct_position = self.chessman.move(col_num, row_num)
        if is_correct_position:
            self.rect.move_ip(
                (col_num - old_col_num) * 80, (old_row_num - row_num) * 80
            )
            self.rect = self.rect.clamp(SCREENRECT)
            self.chessman.chessboard.clear_chessmans_moving_list()
            self.chessman.chessboard.calc_chessmans_moving_list()
            return True
        load_sound("lowtime.mp3").play()
        return False

    def update(self):
        """
        Updates the sprite state (e.g., transparency for selection).
        """
        if self.is_selected:
            if self.is_transparent:
                self.image = self.images[1]
            else:
                self.image = self.images[0]
            self.is_transparent = not self.is_transparent
        else:
            self.image = self.images[0]


def creat_sprite_group(sprite_group, chessmans_hash):
    """
    Creates sprite groups for all pieces on the board.
    """
    for piece in chessmans_hash.values():
        if piece.is_red:
            if isinstance(piece, chessman.Rook):
                images = load_images("red_rook.gif", "transparent.gif")
            elif isinstance(piece, chessman.Cannon):
                images = load_images("red_cannon.gif", "transparent.gif")
            elif isinstance(piece, chessman.Knight):
                images = load_images("red_knight.gif", "transparent.gif")
            elif isinstance(piece, chessman.King):
                images = load_images("red_king.gif", "transparent.gif")
            elif isinstance(piece, chessman.Elephant):
                images = load_images("red_elephant.gif", "transparent.gif")
            elif isinstance(piece, chessman.Mandarin):
                images = load_images("red_mandarin.gif", "transparent.gif")
            else:
                images = load_images("red_pawn.gif", "transparent.gif")
        else:
            if isinstance(piece, chessman.Rook):
                images = load_images("black_rook.gif", "transparent.gif")
            elif isinstance(piece, chessman.Cannon):
                images = load_images("black_cannon.gif", "transparent.gif")
            elif isinstance(piece, chessman.Knight):
                images = load_images("black_knight.gif", "transparent.gif")
            elif isinstance(piece, chessman.King):
                images = load_images("black_king.gif", "transparent.gif")
            elif isinstance(piece, chessman.Elephant):
                images = load_images("black_elephant.gif", "transparent.gif")
            elif isinstance(piece, chessman.Mandarin):
                images = load_images("black_mandarin.gif", "transparent.gif")
            else:
                images = load_images("black_pawn.gif", "transparent.gif")
        if isinstance(piece, chessman.Cannon):
            kill_sound = load_sound("explosion.mp3")
        else:
            kill_sound = load_sound("berserk.mp3")
        chessman_sprite = ChessmanSprite(images, kill_sound, piece)
        sprite_group.add(chessman_sprite)


def select_sprite_from_group(sprite_group, col_num, row_num):
    """
    Selects a sprite from the group based on board coordinates.
    """
    for sprite in sprite_group:
        if sprite.chessman.col_num == col_num and sprite.chessman.row_num == row_num:
            return sprite


def translate_hit_area(cursor):
    """
    Translates screen coordinates to board coordinates (col, row).
    """
    # terminal_x = cursor[0] - left  # Unused
    screen_x, screen_y = cursor[0], cursor[1]
    if screen_x > BOARD_WIDTH:
        return -1, -1
    return screen_x // 80, 9 - screen_y // 80


def draw_button(screen, rect, text, font):
    """
    Draws a button with a shadow and text.
    """
    # Draw shadow
    shadow_rect = rect.copy()
    shadow_rect.move_ip(2, 2)
    pygame.draw.rect(screen, (150, 150, 150), shadow_rect)

    # Draw button
    pygame.draw.rect(screen, (240, 240, 240), rect)
    pygame.draw.rect(screen, (100, 100, 100), rect, 2)  # Border

    # Text
    text_surf = font.render(text, True, (0, 0, 0))
    text_rect = text_surf.get_rect(center=rect.center)
    screen.blit(text_surf, text_rect)


def get_sidebar_rect():
    """
    Calculates the sidebar rectangle area.
    """
    # sidebar_width = 200 # Unused
    sidebar_rect = Rect(BOARD_WIDTH, 0, SIDEBAR_WIDTH, BOARD_HEIGHT)
    return sidebar_rect


def draw_sidebar(
    screen, board, font, small_font, red_time, black_time, game_over_text=""
):
    """
    Draws the sidebar, including the undo button and move history.
    """
    sidebar_rect = get_sidebar_rect()
    SIDEBAR_COLOR = (240, 230, 210)
    pygame.draw.rect(screen, SIDEBAR_COLOR, sidebar_rect)

    # Border line
    pygame.draw.line(
        screen, (0, 0, 0), (BOARD_WIDTH, 0), (BOARD_WIDTH, BOARD_HEIGHT), 2
    )

    # Turn Indicator
    if game_over_text:
        text = game_over_text
        color = (0, 0, 200)  # Blue for result
    elif board.is_red_turn:
        text = "红方走棋"
        color = (200, 0, 0)
    else:
        text = "黑方走棋"
        color = (0, 0, 0)

    def format_time(ms):
        seconds = int(ms / 1000)
        minutes = seconds // 60
        seconds = seconds % 60
        return f"{minutes:02}:{seconds:02}"

    try:
        # Render turn text
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect(center=(BOARD_WIDTH + SIDEBAR_WIDTH // 2, 50))
        screen.blit(text_surface, text_rect)

        # Render Timer
        red_time_str = f"红: {format_time(red_time)}"
        black_time_str = f"黑: {format_time(black_time)}"

        red_time_surf = font.render(
            red_time_str, True, (200, 0, 0)
        )  # Larger font for time
        black_time_surf = font.render(black_time_str, True, (0, 0, 0))

        screen.blit(red_time_surf, (BOARD_WIDTH + 20, 80))
        screen.blit(black_time_surf, (BOARD_WIDTH + 20, 130))

        # Render Move History Header
        history_title = small_font.render("招法记录:", True, (0, 0, 0))
        screen.blit(history_title, (BOARD_WIDTH + 20, 180))

        start_y = 210
        # Show last 15 moves to save space
        recent_moves = board.moves_history[-15:]
        for i, move in enumerate(recent_moves):
            move_idx = len(board.moves_history) - len(recent_moves) + i + 1
            pl_color = (150, 0, 0) if (move_idx % 2 == 1) else (0, 0, 0)
            move_text = small_font.render(f"{move_idx}. {move}", True, pl_color)
            screen.blit(move_text, (BOARD_WIDTH + 20, start_y + i * 24))

    except Exception as e:
        print(f"Sidebar drawing error: {e}")

    # Draw Buttons
    draw_button(screen, BTN_SAVE_RECT, "保存FEN", small_font)
    draw_button(screen, BTN_UNDO_RECT, "悔 棋", small_font)
    draw_button(screen, BTN_RESTART_RECT, "重 开", small_font)


def main(winstyle=0):
    """
    Main function to run the Pygame UI.
    """
    pygame.mixer.init()
    pygame.init()
    bestdepth = pygame.display.mode_ok(SCREENRECT.size, winstyle, 32)
    screen = pygame.display.set_mode(SCREENRECT.size, winstyle, bestdepth)
    pygame.display.set_caption("中国象棋 AI (MyChess Modernized)")

    # Load resources
    try:
        bgdtile = load_image("boardchess.gif")
        load_sound("dong.mp3").play()
    except pygame.error:
        bgdtile = pygame.Surface((80, 80))
        bgdtile.fill((200, 200, 150))

    # Fonts
    font_name = "simhei"
    try:
        font = pygame.font.SysFont(font_name, 36)
        small_font = pygame.font.SysFont(font_name, 24)
    except pygame.error:
        font = pygame.font.SysFont("arial", 36)
        small_font = pygame.font.SysFont("arial", 24)

    # Initialize Game State
    cbd = chessboard.Chessboard("000")
    cbd.init_board()

    # Sprites
    chessmans = pygame.sprite.Group()
    creat_sprite_group(chessmans, cbd.chessmans_hash)
    current_chessman = None
    cbd.calc_chessmans_moving_list()

    framerate = pygame.time.Clock()

    # Timer state
    red_time = 0
    black_time = 0
    last_tick = pygame.time.get_ticks()

    # Background setup
    background = pygame.Surface(SCREENRECT.size)
    for x in range(0, BOARD_WIDTH, bgdtile.get_width()):
        background.blit(bgdtile, (x, 0))

    game_over_text = ""

    while True:  # Main Loop
        # Check Winner
        winner = cbd.get_winner()
        if winner:
            if winner == "Red":
                game_over_text = "红方胜!"
            elif winner == "Black":
                game_over_text = "黑方胜!"
            elif winner == "Draw":
                game_over_text = "和 棋!"
        else:
            game_over_text = ""

        # Timer Update
        current_tick = pygame.time.get_ticks()
        dt = current_tick - last_tick
        last_tick = current_tick

        if not game_over_text:
            if cbd.is_red_turn:
                red_time += dt
            else:
                black_time += dt

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    mouse_x, mouse_y = pygame.mouse.get_pos()

                    # Check Button Clicks
                    if BTN_SAVE_RECT.collidepoint(mouse_x, mouse_y):
                        fen = cbd.to_fen()
                        print(f"FEN Saved: {fen}")
                        continue

                    if BTN_RESTART_RECT.collidepoint(mouse_x, mouse_y):
                        cbd = chessboard.Chessboard("000")
                        cbd.init_board()
                        chessmans.empty()
                        creat_sprite_group(chessmans, cbd.chessmans_hash)
                        current_chessman = None
                        cbd.calc_chessmans_moving_list()
                        red_time = 0
                        black_time = 0
                        game_over_text = ""
                        print("Restarted.")
                        continue

                    if BTN_UNDO_RECT.collidepoint(mouse_x, mouse_y):
                        if (
                            not game_over_text
                        ):  # Allow undo only if game not over? Or allow to revert checkmate?
                            # Standard is allowing undo even after checkmate to analyze
                            # If game_over_text is set, 'winner' is set. Undo clears it.
                            pass

                        if cbd.undo_move():
                            chessmans.empty()
                            creat_sprite_group(chessmans, cbd.chessmans_hash)
                            current_chessman = None
                            cbd.calc_chessmans_moving_list()
                            game_over_text = ""  # Clear game over state
                            print("Undone.")
                        continue

                    # Board Interaction (Only if Game Not Over)
                    if game_over_text:
                        continue

                    col_num, row_num = translate_hit_area((mouse_x, mouse_y))
                    if col_num < 0:
                        continue

                    chessman_sprite = select_sprite_from_group(
                        chessmans, col_num, row_num
                    )

                    if current_chessman is None and chessman_sprite is not None:
                        if chessman_sprite.chessman.is_red == cbd.is_red_turn:
                            current_chessman = chessman_sprite
                            chessman_sprite.is_selected = True
                    elif current_chessman is not None and chessman_sprite is not None:
                        if chessman_sprite.chessman.is_red == cbd.is_red_turn:
                            current_chessman.is_selected = False
                            current_chessman = chessman_sprite
                            chessman_sprite.is_selected = True
                        else:
                            success = current_chessman.move(col_num, row_num)
                            if success:
                                current_chessman.kill_sound.play()
                                chessmans.remove(chessman_sprite)
                                chessman_sprite.kill()
                                current_chessman.is_selected = False
                                current_chessman = None
                    elif current_chessman is not None and chessman_sprite is None:
                        success = current_chessman.move(col_num, row_num)
                        if success:
                            current_chessman.is_selected = False
                            current_chessman.move_sound.play()
                            current_chessman = None

        framerate.tick(20)

        # Draw everything
        screen.blit(background, (0, 0))  # Redraw background
        chessmans.update()
        chessmans.draw(screen)

        # Draw Sidebar
        draw_sidebar(
            screen, cbd, font, small_font, red_time, black_time, game_over_text
        )

        pygame.display.update()


if __name__ == "__main__":
    main()
