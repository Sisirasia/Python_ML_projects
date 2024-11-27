import streamlit as st

# Initialize the game board
if "board" not in st.session_state:
    st.session_state.board = [""] * 9
if "current_player" not in st.session_state:
    st.session_state.current_player = "X"
if "game_over" not in st.session_state:
    st.session_state.game_over = False

def check_winner(board):
    """Check if there's a winner."""
    win_conditions = [
        (0, 1, 2), (3, 4, 5), (6, 7, 8),  # Rows
        (0, 3, 6), (1, 4, 7), (2, 5, 8),  # Columns
        (0, 4, 8), (2, 4, 6)              # Diagonals
    ]
    for condition in win_conditions:
        if board[condition[0]] == board[condition[1]] == board[condition[2]] and board[condition[0]] != "":
            return board[condition[0]]
    return None

def reset_game():
    """Reset the game."""
    st.session_state.board = [""] * 9
    st.session_state.current_player = "X"
    st.session_state.game_over = False

def make_move(index):
    """Handle a player's move."""
    if not st.session_state.game_over and st.session_state.board[index] == "":
        st.session_state.board[index] = st.session_state.current_player
        winner = check_winner(st.session_state.board)
        if winner:
            st.session_state.game_over = True
            st.success(f"Player {winner} wins!")
        elif "" not in st.session_state.board:
            st.session_state.game_over = True
            st.warning("It's a draw!")
        else:
            # Switch player
            st.session_state.current_player = "O" if st.session_state.current_player == "X" else "X"

# Streamlit UI
st.title("Tic Tac Toe")

# Display the game board
for row in range(3):
    cols = st.columns(3)
    for col in range(3):
        index = row * 3 + col
        with cols[col]:
            if st.button(st.session_state.board[index] or " ", key=f"button-{index}"):
                make_move(index)

# Reset button
if st.button("Restart Game"):
    reset_game()
