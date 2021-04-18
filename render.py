import pyglet

def render(task, viewer, FlightModel):
    class DrawText:
        def __init__(self, label: pyglet.text.Label):
            self.label = label

        def render(self):
            self.label.draw()

    screen_width = 1800
    screen_height = 1000

    world_width = 500
    if task == "take-off":
        world_width = 5000
        world_height = 200
    else:
        world_width = 500
        world_height = LEVEL_TARGET * 1.5

    scale = screen_width / world_width
    scale_y = screen_height / world_height
    carty = 100  # TOP OF CART

    cartwidth = 100.0
    cartheight = 20.0

    if viewer is None:
        from gym.envs.classic_control import rendering

        viewer = rendering.Viewer(screen_width, screen_height)

        l, r, t, b = (
            -cartwidth / 2,
            cartwidth / 2,
            cartheight / 2,
            -cartheight / 2,
        )

        axleoffset = cartheight / 4.0
        # cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
        cart = rendering.Image("A320_R.png", 300, 100)
        # cart = rendering.Image()

        self.carttrans = rendering.Transform()
        cart.add_attr(self.carttrans)
        viewer.add_geom(cart)
        if task == "take-off":
            self.track = rendering.Line((0, carty), (screen_width, carty))
        else:
            self.track = rendering.Line(
                (0, LEVEL_TARGET * scale_y + cartheight * 1.1 + (100 - cartheight)),
                (
                    screen_width,
                    LEVEL_TARGET * scale_y + cartheight * 1.1 + (100 - cartheight),
                ),
            )
        self.track.set_color(0, 0, 0)
        viewer.add_geom(self.track)
        self.transform = rendering.Transform()
    if task == "take-off":
        x = FlightModel.Pos[0]
    else:
        x = 250
    y = FlightModel.Pos[1]
    cartx = x * scale + cartwidth * 1.1  # MIDDLE OF CART
    carty = y * scale_y + cartheight * 1.1 + (100 - cartheight)  # MIDDLE OF CART
    self.carttrans.set_translation(cartx, carty)
    self.carttrans.set_rotation(self.FlightModel.theta)
    return viewer
