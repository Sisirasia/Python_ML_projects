import turtle

# Set up the screen
screen = turtle.Screen()
screen.title("Breakout Game")
screen.bgcolor("black")
screen.setup(width=800, height=600)
screen.tracer(0)

# Paddle
paddle = turtle.Turtle()
paddle.speed(0)
paddle.shape("square")
paddle.color("white")
paddle.shapesize(stretch_wid=1, stretch_len=6)
paddle.penup()
paddle.goto(0, -250)

# Ball
ball = turtle.Turtle()
ball.speed(40)
ball.shape("circle")
ball.color("red")
ball.penup()
ball.goto(0, -200)
ball.dx = 2
ball.dy = 2

# Bricks
bricks = []
brick_colors = ["red", "orange", "yellow", "green", "blue"]
y_pos = 250

for color in brick_colors:
    for x in range(-350, 400, 70):  # Spacing between bricks
        brick = turtle.Turtle()
        brick.speed(0)
        brick.shape("square")
        brick.color(color)
        brick.shapesize(stretch_wid=1, stretch_len=3)
        brick.penup()
        brick.goto(x, y_pos)
        bricks.append(brick)
    y_pos -= 30

# Score
score = 0
score_display = turtle.Turtle()
score_display.speed(0)
score_display.color("white")
score_display.penup()
score_display.hideturtle()
score_display.goto(0, 260)
score_display.write(f"Score: {score}", align="center", font=("Courier", 24, "normal"))

# Paddle movement
def paddle_left():
    x = paddle.xcor()
    if x > -350:
        paddle.setx(x - 20)

def paddle_right():
    x = paddle.xcor()
    if x < 350:
        paddle.setx(x + 20)

# Keyboard bindings
screen.listen()
screen.onkeypress(paddle_left, "Left")
screen.onkeypress(paddle_right, "Right")

# Game loop
while True:
    screen.update()

    # Move the ball
    ball.setx(ball.xcor() + ball.dx)
    ball.sety(ball.ycor() + ball.dy)

    # Border collision (left/right)
    if ball.xcor() > 390 or ball.xcor() < -390:
        ball.dx *= -1

    # Border collision (top)
    if ball.ycor() > 290:
        ball.dy *= -1

    # Paddle collision
    if (ball.ycor() > -240 and ball.ycor() < -230) and (
        ball.xcor() > paddle.xcor() - 60 and ball.xcor() < paddle.xcor() + 60
    ):
        ball.sety(-230)
        ball.dy *= -1

    # Bottom border collision (game over)
    if ball.ycor() < -290:
        score_display.clear()
        score_display.write("Game Over", align="center", font=("Courier", 24, "normal"))
        break

    # Brick collision
    for brick in bricks:
        if (
            ball.ycor() > brick.ycor() - 10
            and ball.ycor() < brick.ycor() + 10
            and ball.xcor() > brick.xcor() - 35
            and ball.xcor() < brick.xcor() + 35
        ):
            bricks.remove(brick)
            brick.goto(1000, 1000)  # Move off-screen
            ball.dy *= -1
            score += 10
            score_display.clear()
            score_display.write(f"Score: {score}", align="center", font=("Courier", 24, "normal"))
