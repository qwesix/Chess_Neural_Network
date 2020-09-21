class Pieces:
    empty = 0
    wKing = 1
    wQueen = 2
    wRook = 3
    wBishop = 4
    wKnight = 5
    wPawn = 6
    bKing = -1
    bQueen = -2
    bRook = -3
    bBishop = -4
    bKnight = -5
    bPawn = -6

    switcher = {
        0: "[ ]",
        1: "wKi",
        2: "wQu",
        3: "wRo",
        4: "wBi",
        5: "wKn",
        6: "wPa",
        -1: "bKi",
        -2: "bQu",
        -3: "bRo",
        -4: "bBi",
        -5: "bKn",
        -6: "bPa"
    }