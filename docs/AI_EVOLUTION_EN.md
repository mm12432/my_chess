# From a Regular Board-Game Program to an AI Chess System

This document is written for two groups of readers:

- People who want to build a playable board-game program but do not yet have a full engineering roadmap
- People who are curious about AI, but do not have much deep learning background beyond basic ideas like linear regression, perceptrons, and neural networks

This is not meant to overwhelm readers with terminology at the start. It first explains how a normal chess program is built, then how that program gradually evolves into an AI chess system.

## 1. Start by Separating Three Layers

A complete chess project can usually be split into three layers:

1. Rules layer
2. Interface layer
3. AI layer

### 1.1 Rules Layer

The rules layer answers questions like:

- What does the current board position look like?
- Where is each piece?
- Can this piece move to that square?
- Whose turn is it after the move?
- Who wins, who loses, or is the game drawn?

Plain-language explanation:

The rules layer is the "physics" and "law system" of the project.

In this project, the rules layer mainly lives in:

- `chess_core/chessboard.py`
- `chess_core/chessman.py`
- `chess_core/point.py`

### 1.2 Interface Layer

The interface layer answers questions like:

- Where did the user click?
- How should the board be displayed?
- How should the screen refresh after a move?
- How should command-line input be handled?

Plain-language explanation:

The interface layer is how humans communicate with the program.

In this project, the interface layer mainly lives in:

- `chess_ui/win_game.py`
- `chess_ui/cli_game.py`

### 1.3 AI Layer

The AI layer answers one core question:

- If it is the computer's turn, which move should it choose?

Plain-language explanation:

The AI layer does not define the rules. It makes decisions within the rules.

## 2. How a Regular Chess Program Is Built

Before talking about AI, it is better to first build a program that humans can play.

### 2.1 Step One: Define the Board State

The program must be able to represent the current position.

The basic data usually includes:

- Board size
- Whether each square contains a piece
- The type of each piece
- The color of each piece
- Which side is to move

Plain-language explanation:

The program must first know what the board currently looks like. Otherwise nothing else can be judged correctly.

In this project, `Chessboard` is the core state container.

### 2.2 Step Two: Define Piece Rules

Each piece has its own movement rules:

- How a rook moves
- How a knight moves
- How a cannon moves
- How an elephant moves
- How an advisor moves
- How the king moves
- How a pawn moves

The program must be able to generate candidate moves based on the piece type and current square.

Plain-language explanation:

This step answers:

- Where can this piece move in principle?
- Is the path blocked?
- Does the move go out of bounds?
- Does it violate any special rule?

In this project, this logic is mainly in `Chessman` and its subclasses.

### 2.3 Step Three: Determine Legal Moves

There is an important distinction:

- Candidate moves: moves that seem geometrically possible
- Legal moves: moves that are actually allowed by the rules

Plain-language explanation:

A target square may look reachable from a piece's movement pattern, but it is not legal if it contains a friendly piece or violates a rule.

So a chess program usually needs the ability to:

- List all legal moves from the current position

This is a shared foundation for both normal chess programs and AI systems.

### 2.4 Step Four: Execute a Move

Executing a move usually means doing several things:

1. Remove the piece from its original square
2. Capture an opposing piece if one is on the destination square
3. Place the moving piece on the destination square
4. Record the move
5. Change the side to move
6. Update win/loss state

Plain-language explanation:

"Making a move" is not just changing one coordinate. It is a full state update.

### 2.5 Step Five: Detect Game End

The program must also detect:

- Whether the king has been captured
- Whether a stalemate-like condition has occurred
- Whether repetition-related rules have been triggered

Plain-language explanation:

If the program cannot tell when the game is over, it is not yet a complete chess program.

### 2.6 Step Six: Connect the Interface

Once the rules layer is stable, the interface layer can be added much more cleanly.

The interface layer should usually do only these things:

- Read user input
- Ask the rules layer to attempt a move
- Refresh the display

Plain-language explanation:

The interface layer should not be responsible for understanding the rules. It should simply pass human actions into the rules layer.

## 3. Why the Rules Layer and Interface Layer Should Be Separated

This is a very important engineering idea.

If rules and interface code are tightly mixed together, many problems appear:

- Testing becomes difficult
- AI becomes hard to attach
- Replacing the interface becomes expensive
- Search and simulation become awkward

Plain-language explanation:

If "whether a move is legal" depends on a button, an image, or a sprite object, AI code becomes much harder to build on top of it.

A better structure is:

- The rules layer exists independently
- The interface layer only calls into the rules layer

## 4. How a Regular Chess Program Evolves into an AI Chess System

This is the key transition.

Many people think AI is just one extra function that returns a move. In reality, it requires more structure than that.

To let the program play on its own, we usually need to add several capabilities.

### 4.1 New Capability One: List All Legal Actions

Term: `legal moves`

Plain-language explanation:

The program must know all the moves available in the current position.

Why this matters:

- A human player can look at the board and think
- A computer program needs the allowed choices to be made explicit

In this project, we added:

- `get_legal_moves()`

Its purpose is:

- No dependence on the interface
- Directly list all playable moves from the rules layer

### 4.2 New Capability Two: Represent One Move as a Standard Object

Term: `Move`

Plain-language explanation:

Package "one move" into a standard data structure instead of scattering it across mouse clicks, screen coordinates, and temporary variables.

A move object usually contains at least:

- Source square
- Destination square
- Optionally, the piece identifier

Why this matters:

- AI should not depend on GUI click events
- Search algorithms repeatedly generate and apply moves
- Training data also needs a consistent move representation

### 4.3 New Capability Three: Copy the Board

Term: `clone`

Plain-language explanation:

Make a copy of the current position and test moves on the copy, without changing the real game.

Why this matters:

- A human only needs to look at the current board
- A computer needs to ask "what happens if I try this move?"

Without the ability to copy board state, AI experimentation would corrupt the real game.

### 4.4 New Capability Four: Apply a Move Programmatically

Term: `apply_move(move)`

Plain-language explanation:

Instead of using mouse clicks, let the program pass in a move directly and ask the board to execute it.

Why this matters:

- AI eventually chooses a move object
- The board must accept a move selected by the program

In this project, we added:

- `apply_move(move)`

It reuses the existing move logic, so current gameplay behavior remains intact.

## 5. Why These Changes Do Not Break Existing Gameplay

Because this kind of work is "adding interfaces without changing the game rules."

Plain-language explanation:

The old human-play path still exists:

- The user clicks
- The interface calls the existing move logic
- The board updates

Now there is simply an additional AI path:

- The AI asks for legal moves
- The AI chooses one
- The AI calls `apply_move`

So:

- The old path is preserved
- The new path is added

This is a compatibility-friendly refactor.

## 6. The Next Step from "Playable" to "Able to Think"

At this point, the program has become usable by an AI, but it is not yet an AI itself.

The next stage is the "how does the computer think?" part.

### 6.1 What Is a Model?

Term: `model`

Plain-language explanation:

A model is a system that takes numeric input and produces a judgment. A neural network is one kind of model.

In a chess AI, the model usually does two things:

- Look at the current position and judge which moves deserve attention
- Look at the current position and judge which side is more likely to win

### 6.2 What Is Input Encoding?

Term: `encoding`

Plain-language explanation:

A neural network cannot understand Python objects directly. It only understands numbers.

So we must translate the board into numeric arrays.

For example:

- Where red rooks are
- Where black cannons are
- Which side is to move

All of that must become numeric planes.

### 6.3 What Is the Action Space?

Term: `action space`

Plain-language explanation:

When AI outputs a decision, it needs a fixed move dictionary.

Why:

- A neural network must have a fixed output size
- But the number of legal moves in a position changes from turn to turn

The common solution:

- Define the full list of all possible moves
- In the current position, only use the legal subset

That design problem is called action-space design.

### 6.4 What Is Policy?

Term: `policy`

Plain-language explanation:

A policy does not just say "this is the single correct move." Instead, it gives preference levels across many moves.

For example:

- This move looks more promising
- That move is also possible, but less attractive

A policy output is usually a distribution, not a single answer.

### 6.5 What Is Value?

Term: `value`

Plain-language explanation:

Value means "how good is this position for one side overall?"

It is not the material value of one piece. It is the overall game outlook.

### 6.6 What Is Search?

Term: `search`

Plain-language explanation:

Search means the program mentally explores future move sequences without damaging the real board state.

It repeatedly asks:

- If I play this move, how might the opponent respond?
- Then how might I respond?
- Which branch looks better overall?

Humans do this mentally too; the computer simply turns it into a procedure.

### 6.7 What Is MCTS?

Term: `MCTS`, Monte Carlo Tree Search

Plain-language explanation:

This is a search method that expands promising lines more deeply instead of brute-forcing everything equally.

It can cooperate with a neural network:

- The neural network suggests which moves look promising
- The search checks those moves more deeply

## 7. Why Not Build AI First and Fix Rules Later?

Because the true foundation of chess AI is not "the neural network." It is "a reliable rule system."

If the rules layer is unstable, many problems follow:

- Training data becomes unreliable
- Search results become unreliable
- The model learns from flawed game logic

Plain-language explanation:

No matter how advanced the AI is, it cannot be built on incorrect board logic.

So the usual correct order is:

1. Make the rules correct
2. Refactor the rules into AI-friendly interfaces
3. Then attach models and training systems

## 8. Where This Project Currently Stands

At the time this document is written, the project already has:

- A working rules layer
- GUI and CLI gameplay
- FEN support
- Move notation and board hashing
- AI-facing basic interfaces:
  - `Move`
  - `get_legal_moves()`
  - `apply_move()`
  - `clone()`

This means the project has already evolved:

- From "only humans can play it"
- To "the rules layer can support AI integration"

What it still does not yet have is:

- Action-space indexing
- Numeric board encoding
- Neural network models
- Search algorithms
- Self-play training flow

## 9. A Beginner-Friendly Overall Roadmap

If you want to build a similar project, the following learning and implementation order is recommended.

### Stage One: A Regular Playable Chess Program

First implement:

1. Board state
2. Piece rules
3. Legal move generation
4. Move execution
5. Win/loss detection
6. GUI or CLI

Goal:

- Humans can play chess with the program correctly

### Stage Two: Prepare for AI

Then add:

1. Move objects
2. Legal move enumeration
3. Board cloning
4. Programmatic move application

Goal:

- The program becomes usable not only by humans but also by computer decision logic

### Stage Three: Real AI Integration

Then implement:

1. Action space
2. Board encoding
3. Neural network model
4. Search
5. Self-play
6. Training loop

Goal:

- The program begins to develop the ability to choose moves on its own

## 10. Final One-Sentence Summary

A regular chess program solves:

- Can the game be played correctly?

An AI chess system solves:

- Among all legal moves, which one is the most worth playing?

The first is a rules problem. The second is a decision problem.

So evolving a chess project from "playable" to "able to think" usually does not mean rewriting everything. It means:

- First make the rules solid
- Then refactor them into search- and learning-friendly interfaces
- Finally connect the model and training system

That is the basic path by which a chess project grows from "something you can play" into "something that can think."
