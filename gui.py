import pygame as pg
import pygame_gui
import PokeDex


def make_gui(ally, enemy):
    dex = PokeDex.makeDex()
    pg.init()
    pg.display.set_caption('Pokebot')
    window_surface = pg.display.set_mode ((800, 600))
    background = pg.Surface((800, 600))
    background.fill(pg.Color('#202020'))
    pg.display.set_icon(pg.image.load('Icon.png'))
    manager = pygame_gui.UIManager((800, 600))

    # Ally
    ally_pkmn = pg.image.load('Icons/' + str(ally) + '.png')
    ally_pkmn = pg.transform.flip(ally_pkmn, True, False)
    case_two_types = False
    case_two_types_index = 0
    for i in range(len(dex[ally-1].type)):
        if dex[ally-1].type[i] == '/':
            case_two_types = True
            case_two_types_index = i
    if not case_two_types:
        t1 = pg.image.load('Icons/' + dex[ally-1].type + '.gif')
    else:
        t1 = pg.image.load('Icons/' + dex[ally - 1].type[:case_two_types_index] + '.gif')
        t2 = pg.image.load('Icons/' + dex[ally - 1].type[case_two_types_index+1:] + '.gif')

    pygame_gui.elements.UIButton(relative_rect=pg.Rect((35, 10), (120, 30)), text='Ally', manager=manager)
    pygame_gui.elements.UIButton(relative_rect=pg.Rect((35, 130), (120, 30)), text='[' + dex[ally-1].name + "]", manager=manager)

    # Enemy
    enemy_pkmn = pg.image.load('Icons/' + str(enemy) + '.png')

    pygame_gui.elements.UIButton(relative_rect=pg.Rect((320, 10), (120, 30)), text='Enemy', manager=manager)
    pygame_gui.elements.UIButton(relative_rect=pg.Rect((320, 130), (120, 30)), text='[' + dex[enemy - 1].name + "]", manager=manager)

    case_two_types = False
    case_two_types_index = 0
    for i in range(len(dex[enemy - 1].type)):
        if dex[enemy - 1].type[i] == '/':
            case_two_types = True
            case_two_types_index = i
    if not case_two_types:
        t1_enemy = pg.image.load('Icons/' + dex[enemy - 1].type + '.gif')
    else:
        t1_enemy = pg.image.load('Icons/' + dex[enemy - 1].type[:case_two_types_index] + '.gif')
        t2_enemy = pg.image.load('Icons/' + dex[enemy - 1].type[case_two_types_index + 1:] + '.gif')

    # hello_button = pygame_gui.elements.UIButton(relative_rect=pg.Rect((350, 275), (100, 50)), text='Say Hello', manager=manager)
    clk = pg.time.Clock()
    running = True

    def show_image(f, x, y):
        window_surface.blit(f, (x, y))

    while running:
        time_delta = clk.tick(60)/1000.0
        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False

            if event.type == pg.USEREVENT:
                if event.user_type == pygame_gui.UI_BUTTON_PRESSED:
                    print('Pushed Button!')
                    # if event.ui_element == hello_button:
                        # print('Hello World!')

            manager.process_events(event)

        manager.update(time_delta)

        window_surface.blit(background, (0, 0))
        show_image(ally_pkmn, 60, 50)
        show_image(enemy_pkmn, 345, 50)
        show_image(t1, 45, 160)
        show_image(t2, 95, 160)
        show_image(t1_enemy, 330, 160)
        show_image(t2_enemy, 380, 160)
        manager.draw_ui(window_surface)
        pg.display.update()

