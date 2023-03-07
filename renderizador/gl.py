#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

# pylint: disable=invalid-name

"""
Biblioteca Gráfica / Graphics Library.

Desenvolvido por: <SEU NOME AQUI>
Disciplina: Computação Gráfica
Data: <DATA DE INÍCIO DA IMPLEMENTAÇÃO>
"""

import time  # Para operações com tempo

import gpu  # Simula os recursos de uma GPU

import numpy as np

import math

# transformation_global = None


class GL:
    """Classe que representa a biblioteca gráfica (Graphics Library)."""

    width = 800  # largura da tela
    height = 600  # altura da tela
    near = 0.01  # plano de corte próximo
    far = 1000  # plano de corte distante

    @staticmethod
    def setup(width, height, near=0.01, far=1000):
        """Definr parametros para câmera de razão de aspecto, plano próximo e distante."""
        GL.width = width
        GL.height = height
        GL.near = near
        GL.far = far
        GL.transformation_matrix = None
        GL.look_at = None

    @staticmethod
    def polypoint2D(point, colors):
        """Função usada para renderizar Polypoint2D."""
        # Nessa função você receberá pontos no parâmetro point, esses pontos são uma lista
        # de pontos x, y sempre na ordem. Assim point[0] é o valor da coordenada x do
        # primeiro ponto, point[1] o valor y do primeiro ponto. Já point[2] é a
        # coordenada x do segundo ponto e assim por diante. Assuma a quantidade de pontos
        # pelo tamanho da lista e assuma que sempre vira uma quantidade par de valores.
        # O parâmetro colors é um dicionário com os tipos cores possíveis, para o Polypoint2D
        # você pode assumir o desenho dos pontos com a cor emissiva (emissiveColor).

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("Polypoint2D : pontos = {0}".format(point))  # imprime no terminal pontos
        print(
            "Polypoint2D : colors = {0}".format(colors)
        )  # imprime no terminal as cores
        emissive_colors = colors["emissiveColor"]
        e = [0, 0, 0]
        for i in range(0, len(emissive_colors)):
            e[i] = int(emissive_colors[i]) * 255
        counter = 0

        while counter < len(point):
            gpu.GPU.draw_pixels(
                [int(point[counter]), int(point[counter + 1])],
                gpu.GPU.RGB8,
                [int(e[0]), int(e[1]), int(e[2])],
            )
            counter += 2
        # Exemplo:
        # gpu.GPU.set_pixel(3, 1, 255, 0, 0) altera um pixel da imagem (u, v, r, g, b)
        # cuidado com as cores, o X3D especifica de (0,1) e o Framebuffer de (0,255)

        # Exemplo:
        pos_x = GL.width // 2
        pos_y = GL.height // 2
        gpu.GPU.draw_pixels(
            [pos_x, pos_y], gpu.GPU.RGB8, [255, 0, 0]
        )  # altera pixel (u, v, tipo, r, g, b)
        # cuidado com as cores, o X3D especifica de (0,1) e o Framebuffer de (0,255)

    @staticmethod
    def polyline2D(lineSegments, colors):
        """Função usada para renderizar Polyline2D."""
        # Nessa função você receberá os pontos de uma linha no parâmetro lineSegments, esses
        # pontos são uma lista de pontos x, y sempre na ordem. Assim point[0] é o valor da
        # coordenada x do primeiro ponto, point[1] o valor y do primeiro ponto. Já point[2] é
        # a coordenada x do segundo ponto e assim por diante. Assuma a quantidade de pontos
        # pelo tamanho da lista. A quantidade mínima de pontos são 2 (4 valores), porém a
        # função pode receber mais pontos para desenhar vários segmentos. Assuma que sempre
        # vira uma quantidade par de valores.
        # O parâmetro colors é um dicionário com os tipos cores possíveis, para o Polyline2D
        # você pode assumir o desenho das linhas com a cor emissiva (emissiveColor).

        emissive_colors = colors["emissiveColor"]
        e = [0, 0, 0]
        for i in range(0, len(emissive_colors)):
            e[i] = int(emissive_colors[i]) * 255

        dX = lineSegments[2] - lineSegments[0]
        dY = lineSegments[3] - lineSegments[1]
        if dX == 0 or dY == 0:
            s = 0
        else:
            s = dY / dX

        if s < 1 and s > 0:
            v = lineSegments[1]
            begin = int(lineSegments[0])
            end = int(lineSegments[2])
            step = 1

            for x in range(begin, end, step):
                gpu.GPU.draw_pixels(
                    [int(x), int(round(v))],
                    gpu.GPU.RGB8,
                    [int(e[0]), int(e[1]), int(e[2])],
                )
                v += s

        elif s > 1:
            s = dX / dY
            if dX < 0:
                v = lineSegments[2]
                begin = int(lineSegments[3])
                end = int(lineSegments[1])
            else:
                v = lineSegments[0]
                begin = int(lineSegments[1])
                end = int(lineSegments[3])
            for y in range(begin, end):
                gpu.GPU.draw_pixels(
                    [int(round(v)), int(y)],
                    gpu.GPU.RGB8,
                    [int(e[0]), int(e[1]), int(e[2])],
                )

                v += s
        elif s < -1:
            s = dX / dY
            v = lineSegments[2]
            for y in range(int(lineSegments[3]), int(lineSegments[1])):
                gpu.GPU.draw_pixels(
                    [int(round(v)), int(y)],
                    gpu.GPU.RGB8,
                    [int(e[0]), int(e[1]), int(e[2])],
                )
                v += s
        elif s < 0 and s > -1:
            v = lineSegments[1]

            for x in range(int(lineSegments[2]), int(lineSegments[0])):
                gpu.GPU.draw_pixels(
                    [int(x), int(round(v))],
                    gpu.GPU.RGB8,
                    [int(e[0]), int(e[1]), int(e[2])],
                )
                v += s
        else:
            if dX == 0:
                v = lineSegments[0]
                for y in range(int(lineSegments[1]), int(lineSegments[3])):
                    gpu.GPU.draw_pixels(
                        [int(round(v)), int(y)],
                        gpu.GPU.RGB8,
                        [int(e[0]), int(e[1]), int(e[2])],
                    )
            else:
                v = lineSegments[1]
                for x in range(int(lineSegments[0]), int(lineSegments[2])):
                    gpu.GPU.draw_pixels(
                        [int(x), int(round(v))],
                        gpu.GPU.RGB8,
                        [int(e[0]), int(e[1]), int(e[2])],
                    )

    @staticmethod
    def triangleSet2D(vertices, colors):
        """Função usada para renderizar TriangleSet2D."""
        # Nessa função você receberá os vertices de um triângulo no parâmetro vertices,
        # esses pontos são uma lista de pontos x, y sempre na ordem. Assim point[0] é o
        # valor da coordenada x do primeiro ponto, point[1] o valor y do primeiro ponto.
        # Já point[2] é a coordenada x do segundo ponto e assim por diante. Assuma que a
        # quantidade de pontos é sempre multiplo de 3, ou seja, 6 valores ou 12 valores, etc.
        # O parâmetro colors é um dicionário com os tipos cores possíveis, para o TriangleSet2D
        # você pode assumir o desenho das linhas com a cor emissiva (emissiveColor).
        print("TriangleSet2D : vertices = {0}".format(vertices))  # imprime no terminal
        emissive_colors = colors["emissiveColor"]
        e = [0, 0, 0]
        for i in range(0, len(emissive_colors)):
            e[i] = int(emissive_colors[i]) * 255

        minX = int(min([vertices[0], vertices[2], vertices[4]]))
        maxX = int(max([vertices[0], vertices[2], vertices[4]]))
        minY = int(min([vertices[1], vertices[3], vertices[5]]))
        maxY = int(max([vertices[1], vertices[3], vertices[5]]))

        def L(x0, y0, x1, y1, x, y):
            return (y1 - y0) * x - (x1 - x0) * y + y0 * (x1 - x0) - x0 * (y1 - y0)

        for x in range(minX, maxX + 1):
            for y in range(minY, maxY + 1):
                L1 = L(vertices[0], vertices[1], vertices[2], vertices[3], x, y)
                L2 = L(vertices[2], vertices[3], vertices[4], vertices[5], x, y)
                L3 = L(vertices[4], vertices[5], vertices[0], vertices[1], x, y)
                if L1 >= 0 and L2 >= 0 and L3 >= 0:
                    gpu.GPU.draw_pixels(
                        [int(x), int(y)],
                        gpu.GPU.RGB8,
                        [int(e[0]), int(e[1]), int(e[2])],
                    )

    @staticmethod
    def triangleSet(point, colors):
        """Função usada para renderizar TriangleSet."""
        # Nessa função você receberá pontos no parâmetro point, esses pontos são uma lista
        # de pontos x, y, e z sempre na ordem. Assim point[0] é o valor da coordenada x do
        # primeiro ponto, point[1] o valor y do primeiro ponto, point[2] o valor z da
        # coordenada z do primeiro ponto. Já point[3] é a coordenada x do segundo ponto e
        # assim por diante.
        # No TriangleSet os triângulos são informados individualmente, assim os três
        # primeiros pontos definem um triângulo, os três próximos pontos definem um novo
        # triângulo, e assim por diante.
        # O parâmetro colors é um dicionário com os tipos cores possíveis, para o TriangleSet
        # você pode assumir o desenho das linhas com a cor emissiva (emissiveColor).

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.

        final_matrix = np.matmul(GL.look_at, GL.transformation_matrix)

        p1 = np.array([[point[0]], [point[1]], [point[2]], [1]])
        p2 = np.array([[point[3]], [point[4]], [point[5]], [1]])
        p3 = np.array([[point[6]], [point[7]], [point[8]], [1]])

        p1 = np.matmul(final_matrix, p1)
        p2 = np.matmul(final_matrix, p2)
        p3 = np.matmul(final_matrix, p3)

        p1 = p1 / p1[3][0]
        p2 = p2 / p2[3][0]
        p3 = p3 / p3[3][0]

        GL.triangleSet2D([p1[0], p1[1], p2[0], p2[1], p3[0], p3[1]], colors)

        print("TriangleSet : pontos = {0}".format(point))  # imprime no terminal pontos
        print(
            "TriangleSet : colors = {0}".format(colors)
        )  # imprime no terminal as cores
        print("transf = {0}".format(GL.transformation_matrix))

        # Exemplo de desenho de um pixel branco na coordenada 10, 10
        gpu.GPU.draw_pixels([10, 10], gpu.GPU.RGB8, [255, 255, 255])  # altera pixel

    @staticmethod
    def calc_rotation(rot):
        qi = rot[0] * (math.sin(math.radians(rot[3]) / 2))
        qj = rot[1] * (math.sin(math.radians(rot[3]) / 2))
        qk = rot[2] * (math.sin(math.radians(rot[3]) / 2))
        qr = math.cos(math.radians(rot[3]) / 2)

        rotation_matrix = np.array(
            [
                [
                    (1 - 2 * ((qj**2) + (qk**2))),
                    (2 * (qi * qj - qk * qr)),
                    (2 * (qi * qk + qj * qr)),
                    0,
                ],
                [
                    (2 * (qi * qj + qk * qr)),
                    (1 - 2 * ((qi**2) + (qk**2))),
                    (2 * (qj * qk - qi * qr)),
                    0,
                ],
                [
                    (2 * (qi * qk - qj * qr)),
                    (2 * (qj * qk + qi * qr)),
                    (1 - 2 * ((qi**2) + (qj**2))),
                    0,
                ],
                [0, 0, 0, 1],
            ]
        )
        return rotation_matrix

    @staticmethod
    def viewpoint(position, orientation, fieldOfView):
        """Função usada para renderizar (na verdade coletar os dados) de Viewpoint."""
        # Na função de viewpoint você receberá a posição, orientação e campo de visão da
        # câmera virtual. Use esses dados para poder calcular e criar a matriz de projeção
        # perspectiva para poder aplicar nos pontos dos objetos geométricos.
        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        position_matrix = np.array(
            [
                [1, 0, 0, -position[0]],
                [0, 1, 0, -position[1]],
                [0, 0, 1, -position[2]],
                [0, 0, 0, 1],
            ]
        )
        orientation_matrix = GL.calc_rotation(orientation).transpose()
        GL.look_at = np.matmul(orientation_matrix, position_matrix)

        multiplicador = GL.height / (((GL.height**2) + (GL.width**2)) ** 0.5)

        FOVy = 2 * math.atan(math.tan(fieldOfView / 2) * multiplicador)

        aspect = GL.width / GL.height

        top = GL.near * math.tan(FOVy)
        bottom = -top
        right = top * aspect
        left = -right

        P = np.array(
            [
                [GL.near / right, 0, 0, 0],
                [0, GL.near / top, 0, 0],
                [
                    0,
                    0,
                    -((GL.far + GL.near) / (GL.far - GL.near)),
                    -2 * (GL.far * GL.near) / (GL.far - GL.near),
                ],
                [0, 0, -1, 0],
            ]
        )

        screen = np.array(
            [
                [GL.width / 2, 0, 0, GL.width / 2],
                [0, -GL.height / 2, 0, GL.height / 2],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )

        GL.look_at = np.matmul(P, GL.look_at)

        GL.look_at = np.matmul(screen, GL.look_at)

        print("Viewpoint : ", end="")
        print("position = {0} ".format(position), end="")
        print("orientation = {0} ".format(orientation), end="")
        print("fieldOfView = {0} ".format(fieldOfView))
        print("Look at = {0}".format(GL.look_at))

    @staticmethod
    def transform_in(translation, scale, rotation):
        """Função usada para renderizar (na verdade coletar os dados) de Transform."""
        # A função transform_in será chamada quando se entrar em um nó X3D do tipo Transform
        # do grafo de cena. Os valores passados são a escala em um vetor [x, y, z]
        # indicando a escala em cada direção, a translação [x, y, z] nas respectivas
        # coordenadas e finalmente a rotação por [x, y, z, t] sendo definida pela rotação
        # do objeto ao redor do eixo x, y, z por t radianos, seguindo a regra da mão direita.
        # Quando se entrar em um nó transform se deverá salvar a matriz de transformação dos
        # modelos do mundo em alguma estrutura de pilha.
        # Escala, rotação, translação

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("Transform : ", end="")
        if translation:
            print(
                "translation = {0} ".format(translation), end=""
            )  # imprime no terminal
            translation_matrix = np.array(
                [
                    [1, 0, 0, translation[0]],
                    [0, 1, 0, translation[1]],
                    [0, 0, 1, translation[2]],
                    [0, 0, 0, 1],
                ]
            )
        if scale:
            print("scale = {0} ".format(scale), end="")  # imprime no terminal
            scale_matrix = np.array(
                [
                    [scale[0], 0, 0, 0],
                    [0, scale[1], 0, 0],
                    [0, 0, scale[2], 0],
                    [0, 0, 0, 1],
                ]
            )
        if rotation:
            print("rotation = {0} ".format(rotation), end="")  # imprime no terminal

        temp = np.matmul(translation_matrix, GL.calc_rotation(rotation))

        GL.transformation_matrix = np.matmul(temp, scale_matrix)

        print("")

    @staticmethod
    def transform_out():
        """Função usada para renderizar (na verdade coletar os dados) de Transform."""
        # A função transform_out será chamada quando se sair em um nó X3D do tipo Transform do
        # grafo de cena. Não são passados valores, porém quando se sai de um nó transform se
        # deverá recuperar a matriz de transformação dos modelos do mundo da estrutura de
        # pilha implementada.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("Saindo de Transform")

    @staticmethod
    def triangleStripSet(point, stripCount, colors):
        """Função usada para renderizar TriangleStripSet."""
        # A função triangleStripSet é usada para desenhar tiras de triângulos interconectados,
        # você receberá as coordenadas dos pontos no parâmetro point, esses pontos são uma
        # lista de pontos x, y, e z sempre na ordem. Assim point[0] é o valor da coordenada x
        # do primeiro ponto, point[1] o valor y do primeiro ponto, point[2] o valor z da
        # coordenada z do primeiro ponto. Já point[3] é a coordenada x do segundo ponto e assim
        # por diante. No TriangleStripSet a quantidade de vértices a serem usados é informado
        # em uma lista chamada stripCount (perceba que é uma lista). Ligue os vértices na ordem,
        # primeiro triângulo será com os vértices 0, 1 e 2, depois serão os vértices 1, 2 e 3,
        # depois 2, 3 e 4, e assim por diante. Cuidado com a orientação dos vértices, ou seja,
        # todos no sentido horário ou todos no sentido anti-horário, conforme especificado.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("TriangleStripSet : pontos = {0} ".format(point), end="")
        for i, strip in enumerate(stripCount):
            print("strip[{0}] = {1} ".format(i, strip), end="")
        print("")
        print(
            "TriangleStripSet : colors = {0}".format(colors)
        )  # imprime no terminal as cores

        # Exemplo de desenho de um pixel branco na coordenada 10, 10
        gpu.GPU.draw_pixels([10, 10], gpu.GPU.RGB8, [255, 255, 255])  # altera pixel

    @staticmethod
    def indexedTriangleStripSet(point, index, colors):
        """Função usada para renderizar IndexedTriangleStripSet."""
        # A função indexedTriangleStripSet é usada para desenhar tiras de triângulos
        # interconectados, você receberá as coordenadas dos pontos no parâmetro point, esses
        # pontos são uma lista de pontos x, y, e z sempre na ordem. Assim point[0] é o valor
        # da coordenada x do primeiro ponto, point[1] o valor y do primeiro ponto, point[2]
        # o valor z da coordenada z do primeiro ponto. Já point[3] é a coordenada x do
        # segundo ponto e assim por diante. No IndexedTriangleStripSet uma lista informando
        # como conectar os vértices é informada em index, o valor -1 indica que a lista
        # acabou. A ordem de conexão será de 3 em 3 pulando um índice. Por exemplo: o
        # primeiro triângulo será com os vértices 0, 1 e 2, depois serão os vértices 1, 2 e 3,
        # depois 2, 3 e 4, e assim por diante. Cuidado com a orientação dos vértices, ou seja,
        # todos no sentido horário ou todos no sentido anti-horário, conforme especificado.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print(
            "IndexedTriangleStripSet : pontos = {0}, index = {1}".format(point, index)
        )
        print(
            "IndexedTriangleStripSet : colors = {0}".format(colors)
        )  # imprime as cores

        # Exemplo de desenho de um pixel branco na coordenada 10, 10
        gpu.GPU.draw_pixels([10, 10], gpu.GPU.RGB8, [255, 255, 255])  # altera pixel

    @staticmethod
    def box(size, colors):
        """Função usada para renderizar Boxes."""
        # A função box é usada para desenhar paralelepípedos na cena. O Box é centrada no
        # (0, 0, 0) no sistema de coordenadas local e alinhado com os eixos de coordenadas
        # locais. O argumento size especifica as extensões da caixa ao longo dos eixos X, Y
        # e Z, respectivamente, e cada valor do tamanho deve ser maior que zero. Para desenha
        # essa caixa você vai provavelmente querer tesselar ela em triângulos, para isso
        # encontre os vértices e defina os triângulos.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("Box : size = {0}".format(size))  # imprime no terminal pontos
        print("Box : colors = {0}".format(colors))  # imprime no terminal as cores

        # Exemplo de desenho de um pixel branco na coordenada 10, 10
        gpu.GPU.draw_pixels([10, 10], gpu.GPU.RGB8, [255, 255, 255])  # altera pixel

    @staticmethod
    def indexedFaceSet(
        coord,
        coordIndex,
        colorPerVertex,
        color,
        colorIndex,
        texCoord,
        texCoordIndex,
        colors,
        current_texture,
    ):
        """Função usada para renderizar IndexedFaceSet."""
        # A função indexedFaceSet é usada para desenhar malhas de triângulos. Ela funciona de
        # forma muito simular a IndexedTriangleStripSet porém com mais recursos.
        # Você receberá as coordenadas dos pontos no parâmetro cord, esses
        # pontos são uma lista de pontos x, y, e z sempre na ordem. Assim coord[0] é o valor
        # da coordenada x do primeiro ponto, coord[1] o valor y do primeiro ponto, coord[2]
        # o valor z da coordenada z do primeiro ponto. Já coord[3] é a coordenada x do
        # segundo ponto e assim por diante. No IndexedFaceSet uma lista de vértices é informada
        # em coordIndex, o valor -1 indica que a lista acabou.
        # A ordem de conexão será de 3 em 3 pulando um índice. Por exemplo: o
        # primeiro triângulo será com os vértices 0, 1 e 2, depois serão os vértices 1, 2 e 3,
        # depois 2, 3 e 4, e assim por diante.
        # Adicionalmente essa implementação do IndexedFace aceita cores por vértices, assim
        # se a flag colorPerVertex estiver habilitada, os vértices também possuirão cores
        # que servem para definir a cor interna dos poligonos, para isso faça um cálculo
        # baricêntrico de que cor deverá ter aquela posição. Da mesma forma se pode definir uma
        # textura para o poligono, para isso, use as coordenadas de textura e depois aplique a
        # cor da textura conforme a posição do mapeamento. Dentro da classe GPU já está
        # implementadado um método para a leitura de imagens.

        # Os prints abaixo são só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("IndexedFaceSet : ")
        if coord:
            print("\tpontos(x, y, z) = {0}, coordIndex = {1}".format(coord, coordIndex))
        print("colorPerVertex = {0}".format(colorPerVertex))
        if colorPerVertex and color and colorIndex:
            print("\tcores(r, g, b) = {0}, colorIndex = {1}".format(color, colorIndex))
        if texCoord and texCoordIndex:
            print(
                "\tpontos(u, v) = {0}, texCoordIndex = {1}".format(
                    texCoord, texCoordIndex
                )
            )
        if current_texture:
            image = gpu.GPU.load_texture(current_texture[0])
            print("\t Matriz com image = {0}".format(image))
            print("\t Dimensões da image = {0}".format(image.shape))
        print(
            "IndexedFaceSet : colors = {0}".format(colors)
        )  # imprime no terminal as cores

        # Exemplo de desenho de um pixel branco na coordenada 10, 10
        gpu.GPU.draw_pixels([10, 10], gpu.GPU.RGB8, [255, 255, 255])  # altera pixel

    @staticmethod
    def sphere(radius, colors):
        """Função usada para renderizar Esferas."""
        # A função sphere é usada para desenhar esferas na cena. O esfera é centrada no
        # (0, 0, 0) no sistema de coordenadas local. O argumento radius especifica o
        # raio da esfera que está sendo criada. Para desenha essa esfera você vai
        # precisar tesselar ela em triângulos, para isso encontre os vértices e defina
        # os triângulos.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print(
            "Sphere : radius = {0}".format(radius)
        )  # imprime no terminal o raio da esfera
        print("Sphere : colors = {0}".format(colors))  # imprime no terminal as cores

    @staticmethod
    def navigationInfo(headlight):
        """Características físicas do avatar do visualizador e do modelo de visualização."""
        # O campo do headlight especifica se um navegador deve acender um luz direcional que
        # sempre aponta na direção que o usuário está olhando. Definir este campo como TRUE
        # faz com que o visualizador forneça sempre uma luz do ponto de vista do usuário.
        # A luz headlight deve ser direcional, ter intensidade = 1, cor = (1 1 1),
        # ambientIntensity = 0,0 e direção = (0 0 −1).

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print(
            "NavigationInfo : headlight = {0}".format(headlight)
        )  # imprime no terminal

    @staticmethod
    def directionalLight(ambientIntensity, color, intensity, direction):
        """Luz direcional ou paralela."""
        # Define uma fonte de luz direcional que ilumina ao longo de raios paralelos
        # em um determinado vetor tridimensional. Possui os campos básicos ambientIntensity,
        # cor, intensidade. O campo de direção especifica o vetor de direção da iluminação
        # que emana da fonte de luz no sistema de coordenadas local. A luz é emitida ao
        # longo de raios paralelos de uma distância infinita.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("DirectionalLight : ambientIntensity = {0}".format(ambientIntensity))
        print("DirectionalLight : color = {0}".format(color))  # imprime no terminal
        print(
            "DirectionalLight : intensity = {0}".format(intensity)
        )  # imprime no terminal
        print(
            "DirectionalLight : direction = {0}".format(direction)
        )  # imprime no terminal

    @staticmethod
    def pointLight(ambientIntensity, color, intensity, location):
        """Luz pontual."""
        # Fonte de luz pontual em um local 3D no sistema de coordenadas local. Uma fonte
        # de luz pontual emite luz igualmente em todas as direções; ou seja, é omnidirecional.
        # Possui os campos básicos ambientIntensity, cor, intensidade. Um nó PointLight ilumina
        # a geometria em um raio de sua localização. O campo do raio deve ser maior ou igual a
        # zero. A iluminação do nó PointLight diminui com a distância especificada.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("PointLight : ambientIntensity = {0}".format(ambientIntensity))
        print("PointLight : color = {0}".format(color))  # imprime no terminal
        print("PointLight : intensity = {0}".format(intensity))  # imprime no terminal
        print("PointLight : location = {0}".format(location))  # imprime no terminal

    @staticmethod
    def fog(visibilityRange, color):
        """Névoa."""
        # O nó Fog fornece uma maneira de simular efeitos atmosféricos combinando objetos
        # com a cor especificada pelo campo de cores com base nas distâncias dos
        # vários objetos ao visualizador. A visibilidadeRange especifica a distância no
        # sistema de coordenadas local na qual os objetos são totalmente obscurecidos
        # pela névoa. Os objetos localizados fora de visibilityRange do visualizador são
        # desenhados com uma cor de cor constante. Objetos muito próximos do visualizador
        # são muito pouco misturados com a cor do nevoeiro.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("Fog : color = {0}".format(color))  # imprime no terminal
        print("Fog : visibilityRange = {0}".format(visibilityRange))

    @staticmethod
    def timeSensor(cycleInterval, loop):
        """Gera eventos conforme o tempo passa."""
        # Os nós TimeSensor podem ser usados para muitas finalidades, incluindo:
        # Condução de simulações e animações contínuas; Controlar atividades periódicas;
        # iniciar eventos de ocorrência única, como um despertador;
        # Se, no final de um ciclo, o valor do loop for FALSE, a execução é encerrada.
        # Por outro lado, se o loop for TRUE no final de um ciclo, um nó dependente do
        # tempo continua a execução no próximo ciclo. O ciclo de um nó TimeSensor dura
        # cycleInterval segundos. O valor de cycleInterval deve ser maior que zero.

        # Deve retornar a fração de tempo passada em fraction_changed

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print(
            "TimeSensor : cycleInterval = {0}".format(cycleInterval)
        )  # imprime no terminal
        print("TimeSensor : loop = {0}".format(loop))

        # Esse método já está implementado para os alunos como exemplo
        epoch = (
            time.time()
        )  # time in seconds since the epoch as a floating point number.
        fraction_changed = (epoch % cycleInterval) / cycleInterval

        return fraction_changed

    @staticmethod
    def splinePositionInterpolator(set_fraction, key, keyValue, closed):
        """Interpola não linearmente entre uma lista de vetores 3D."""
        # Interpola não linearmente entre uma lista de vetores 3D. O campo keyValue possui
        # uma lista com os valores a serem interpolados, key possui uma lista respectiva de chaves
        # dos valores em keyValue, a fração a ser interpolada vem de set_fraction que varia de
        # zeroa a um. O campo keyValue deve conter exatamente tantos vetores 3D quanto os
        # quadros-chave no key. O campo closed especifica se o interpolador deve tratar a malha
        # como fechada, com uma transições da última chave para a primeira chave. Se os keyValues
        # na primeira e na última chave não forem idênticos, o campo closed será ignorado.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("SplinePositionInterpolator : set_fraction = {0}".format(set_fraction))
        print(
            "SplinePositionInterpolator : key = {0}".format(key)
        )  # imprime no terminal
        print("SplinePositionInterpolator : keyValue = {0}".format(keyValue))
        print("SplinePositionInterpolator : closed = {0}".format(closed))

        # Abaixo está só um exemplo de como os dados podem ser calculados e transferidos
        value_changed = [0.0, 0.0, 0.0]

        return value_changed

    @staticmethod
    def orientationInterpolator(set_fraction, key, keyValue):
        """Interpola entre uma lista de valores de rotação especificos."""
        # Interpola rotações são absolutas no espaço do objeto e, portanto, não são cumulativas.
        # Uma orientação representa a posição final de um objeto após a aplicação de uma rotação.
        # Um OrientationInterpolator interpola entre duas orientações calculando o caminho mais
        # curto na esfera unitária entre as duas orientações. A interpolação é linear em
        # comprimento de arco ao longo deste caminho. Os resultados são indefinidos se as duas
        # orientações forem diagonalmente opostas. O campo keyValue possui uma lista com os
        # valores a serem interpolados, key possui uma lista respectiva de chaves
        # dos valores em keyValue, a fração a ser interpolada vem de set_fraction que varia de
        # zeroa a um. O campo keyValue deve conter exatamente tantas rotações 3D quanto os
        # quadros-chave no key.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("OrientationInterpolator : set_fraction = {0}".format(set_fraction))
        print("OrientationInterpolator : key = {0}".format(key))  # imprime no terminal
        print("OrientationInterpolator : keyValue = {0}".format(keyValue))

        # Abaixo está só um exemplo de como os dados podem ser calculados e transferidos
        value_changed = [0, 0, 1, 0]

        return value_changed

    # Para o futuro (Não para versão atual do projeto.)
    def vertex_shader(self, shader):
        """Para no futuro implementar um vertex shader."""

    def fragment_shader(self, shader):
        """Para no futuro implementar um fragment shader."""
