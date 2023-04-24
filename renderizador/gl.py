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
        GL.transformation_matrix = [
                 [1,0,0,0],
                 [0,1,0,0],
                 [0,0,1,0],
                 [0,0,0,1]]
        GL.zbuffer = np.array(np.ones((height, width)) * np.inf)
        GL.anti_aliasing = False
        GL.stack = []
        GL.look_at = None
        GL.color_per_vertex = False
        GL.transp = 0
        GL.P_mat = None
        GL.is_texture = False
        GL.cur_text = None
        GL.headlight = False
        GL.screen = None

    @staticmethod
    def polypoint2D(point, colors):
        """Função usada para renderizar Polypoint2D."""
        # Nessa função você receberá pontos no parâmetro point, esses pontos são uma lista
        # de pontos x, y sempre na ordem. Assim point[0] é o valor da coordenada x do
        # primeiro ponto, point[1] o valor y do primeiro ponto. Já point[2] é a
        # coordenada x do segundo ponto e assim por diante. Assuma a quantidade de pontos
        # pelo tamanho da lista e assuma que sempre vira uma quantidade par de valores.
        # O parâmetro colors é um dicionário com os tipos cores possíveis, para o Polypoint2D
        # você pode assumir inicialmente o desenho dos pontos com a cor emissiva (emissiveColor).

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.

        emissive_colors = colors["emissiveColor"]
        e = [0, 0, 0]
        for i in range(0, len(emissive_colors)):
            e[i] = int(emissive_colors[i]) * 255
        counter = 0

        while counter < len(point):
            gpu.GPU.draw_pixel(
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
        gpu.GPU.draw_pixel(
            [pos_x, pos_y], gpu.GPU.RGB8, [255, 0, 0]
        )  # altera pixel (u, v, tipo, r, g, b)
        # cuidado com as cores, o X3D especifica de (0,1) e o Framebuffer de (0,255)

        # Exemplo:
        pos_x = GL.width // 2
        pos_y = GL.height // 2
        gpu.GPU.draw_pixel(
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
                gpu.GPU.draw_pixel(
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
                gpu.GPU.draw_pixel(
                    [int(round(v)), int(y)],
                    gpu.GPU.RGB8,
                    [int(e[0]), int(e[1]), int(e[2])],
                )

                v += s
        elif s < -1:
            s = dX / dY
            v = lineSegments[2]
            for y in range(int(lineSegments[3]), int(lineSegments[1])):
                gpu.GPU.draw_pixel(
                    [int(round(v)), int(y)],
                    gpu.GPU.RGB8,
                    [int(e[0]), int(e[1]), int(e[2])],
                )
                v += s
        elif s < 0 and s > -1:
            v = lineSegments[1]

            for x in range(int(lineSegments[2]), int(lineSegments[0])):
                gpu.GPU.draw_pixel(
                    [int(x), int(round(v))],
                    gpu.GPU.RGB8,
                    [int(e[0]), int(e[1]), int(e[2])],
                )
                v += s
        else:
            if dX == 0:
                v = lineSegments[0]
                for y in range(int(lineSegments[1]), int(lineSegments[3])):
                    gpu.GPU.draw_pixel(
                        [int(round(v)), int(y)],
                        gpu.GPU.RGB8,
                        [int(e[0]), int(e[1]), int(e[2])],
                    )
            else:
                v = lineSegments[1]
                for x in range(int(lineSegments[0]), int(lineSegments[2])):
                    gpu.GPU.draw_pixel(
                        [int(x), int(round(v))],
                        gpu.GPU.RGB8,
                        [int(e[0]), int(e[1]), int(e[2])],
                    )
    
    @staticmethod
    def triangleSet2DColorPerVertex(vertices_x_y, colors):
        minX = int(min([vertices_x_y[0], vertices_x_y[2], vertices_x_y[4]]))
        maxX = int(max([vertices_x_y[0], vertices_x_y[2], vertices_x_y[4]]))
        minY = int(min([vertices_x_y[1], vertices_x_y[3], vertices_x_y[5]]))
        maxY = int(max([vertices_x_y[1], vertices_x_y[3], vertices_x_y[5]]))

        def L(x0, y0, x1, y1, x, y):
            return (y1 - y0) * x - (x1 - x0) * y + y0 * (x1 - x0) - x0 * (y1 - y0)
        def is_inside(x_sample, y_sample):
            L1 = L(vertices_x_y[0], vertices_x_y[1], vertices_x_y[2], vertices_x_y[3], x_sample, y_sample)
            L2 = L(vertices_x_y[2], vertices_x_y[3], vertices_x_y[4], vertices_x_y[5], x_sample, y_sample)
            L3 = L(vertices_x_y[4], vertices_x_y[5], vertices_x_y[0], vertices_x_y[1], x_sample, y_sample)
            if L1 >= 0 and L2 >= 0 and L3 >= 0:
                return 1
            else:
                return 0
            
        def calc_area_triangulo (vertice1, vertice2, vertice3, ponto):
            pv1 = [vertice1[0] - ponto[0], vertice1[1] - ponto[1]]
            pv2 = [vertice2[0] - ponto[0], vertice2[1] - ponto[1]]
            pv3 = [vertice3[0] - ponto[0], vertice3[1] - ponto[1]]
            va = np.linalg.norm(np.cross(pv2, pv3))/2
            vb = np.linalg.norm(np.cross(pv1, pv3))/2
            vc = np.linalg.norm(np.cross(pv1, pv2))/2

            area = va + vb + vc

            red = (va*colors[0] + vb*colors[3] + vc*colors[6]) / area
            red = ( red - 0 ) * (255 - 0) / ( 1 - 0 )

            green = (va*colors[1] + vb*colors[4] + vc*colors[7]) / area
            green = ( green - 0 ) * (255 - 0) / ( 1 - 0 )

            blue = (va*colors[2] + vb*colors[5] + vc*colors[8]) / area
            blue = ( blue - 0 ) * (255 - 0) / ( 1 - 0 )

            return red, green, blue
        

        for x in range(minX, maxX + 1):
            for y in range(minY, maxY + 1):
                if GL.anti_aliasing:
                    media_dentro = is_inside(x - 0.25 , y - 0.25) + is_inside(x + 0.25 , y + 0.25) + is_inside(x - 0.25 , y + 0.25) + is_inside(x + 0.25 , y - 0.25)
                    media_dentro = media_dentro/4
                    if(media_dentro > 0):
                        r, g, b = calc_area_triangulo([vertices_x_y[0], vertices_x_y[1]], [vertices_x_y[2], vertices_x_y[3]], [vertices_x_y[4], vertices_x_y[5]], [x,y])
                        gpu.GPU.draw_pixel(
                            [int(x), int(y)],
                            gpu.GPU.RGB8,
                            [int(r*media_dentro * (1 - GL.transp)), int(g*media_dentro  * (1 - GL.transp)), int(b*media_dentro  * (1 - GL.transp))], #With Supersampling
                        )
                else:
                    if is_inside(x, y):
                        r, g, b = calc_area_triangulo([vertices_x_y[0], vertices_x_y[1]], [vertices_x_y[2], vertices_x_y[3]], [vertices_x_y[4], vertices_x_y[5]], [x,y])
                        gpu.GPU.draw_pixel(
                            [int(x), int(y)],
                            gpu.GPU.RGB8,
                            [int(r  * (1 - GL.transp)), int(g  * (1 - GL.transp)), int(b  * (1 - GL.transp))], #With Supersampling
                        )

    @staticmethod
    def triangleSet2DColorPerTexture(vertices_x_y, colors):
        uv_text = colors
        minX = int(min([vertices_x_y[0], vertices_x_y[2], vertices_x_y[4]]))
        maxX = int(max([vertices_x_y[0], vertices_x_y[2], vertices_x_y[4]]))
        minY = int(min([vertices_x_y[1], vertices_x_y[3], vertices_x_y[5]]))
        maxY = int(max([vertices_x_y[1], vertices_x_y[3], vertices_x_y[5]]))

        def L(x0, y0, x1, y1, x, y):
            return (y1 - y0) * x - (x1 - x0) * y + y0 * (x1 - x0) - x0 * (y1 - y0)
        def is_inside(x_sample, y_sample):
            L1 = L(vertices_x_y[0], vertices_x_y[1], vertices_x_y[2], vertices_x_y[3], x_sample, y_sample)
            L2 = L(vertices_x_y[2], vertices_x_y[3], vertices_x_y[4], vertices_x_y[5], x_sample, y_sample)
            L3 = L(vertices_x_y[4], vertices_x_y[5], vertices_x_y[0], vertices_x_y[1], x_sample, y_sample)
            if L1 >= 0 and L2 >= 0 and L3 >= 0:
                return 1
            else:
                return 0
            
        def calc_area_triangulo (vertice1, vertice2, vertice3, ponto, uv_text):
            pv1 = [vertice1[0] - ponto[0], vertice1[1] - ponto[1]]
            pv2 = [vertice2[0] - ponto[0], vertice2[1] - ponto[1]]
            pv3 = [vertice3[0] - ponto[0], vertice3[1] - ponto[1]]
            va = np.linalg.norm(np.cross(pv2, pv3))/2
            vb = np.linalg.norm(np.cross(pv1, pv3))/2
            vc = np.linalg.norm(np.cross(pv1, pv2))/2

            area = va + vb + vc
            
            u = ((va*uv_text[0] + vb*uv_text[2] + vc*uv_text[4]) / area)*(len(GL.cur_text)-1)
            

            v = -((va*uv_text[1] + vb*uv_text[3] + vc*uv_text[5]) / area)*(len(GL.cur_text[0])-1)

            
            red = GL.cur_text[int(v)][int(u)][0]
            green = GL.cur_text[int(v)][int(u)][1]
            blue = GL.cur_text[int(v)][int(u)][2]
           
            return red, green, blue
        

        for x in range(minX, maxX + 1):
            for y in range(minY, maxY + 1):
                if GL.anti_aliasing:
                    media_dentro = is_inside(x - 0.25 , y - 0.25) + is_inside(x + 0.25 , y + 0.25) + is_inside(x - 0.25 , y + 0.25) + is_inside(x + 0.25 , y - 0.25)
                    media_dentro = media_dentro/4
                    if(media_dentro > 0):
                        r, g, b = calc_area_triangulo([vertices_x_y[0], vertices_x_y[1]], [vertices_x_y[2], vertices_x_y[3]], [vertices_x_y[4], vertices_x_y[5]], [x,y], uv_text)
                        gpu.GPU.draw_pixel(
                            [int(x), int(y)],
                            gpu.GPU.RGB8,
                            [int(r*media_dentro), int(g*media_dentro), int(b*media_dentro)],
                        )
                else:
                    if is_inside(x, y):
                        r, g, b = calc_area_triangulo([vertices_x_y[0], vertices_x_y[1]], [vertices_x_y[2], vertices_x_y[3]], [vertices_x_y[4], vertices_x_y[5]], [x,y], uv_text)
                        gpu.GPU.draw_pixel(
                            [int(x), int(y)],
                            gpu.GPU.RGB8,
                            [int(r), int(g), int(b)],
                        )
    @staticmethod
    def check_is_inside_tri(x_sample, y_sample, vertices_x_y):
        def L(x0, y0, x1, y1, x, y):
            return (y1 - y0) * x - (x1 - x0) * y + y0 * (x1 - x0) - x0 * (y1 - y0)

        L1 = L(vertices_x_y[0], vertices_x_y[1], vertices_x_y[2], vertices_x_y[3], x_sample, y_sample)
        L2 = L(vertices_x_y[2], vertices_x_y[3], vertices_x_y[4], vertices_x_y[5], x_sample, y_sample)
        L3 = L(vertices_x_y[4], vertices_x_y[5], vertices_x_y[0], vertices_x_y[1], x_sample, y_sample)
        if L1 >= 0 and L2 >= 0 and L3 >= 0:
            return 1
        else:
            return 0

    @ staticmethod
    def draw_with_anti_aliasing(x, y, vertices_x_y, e):
        media_dentro = GL.check_is_inside_tri(x - 0.25 , y - 0.25, vertices_x_y) + GL.check_is_inside_tri(x + 0.25 , y + 0.25, vertices_x_y) + GL.check_is_inside_tri(x - 0.25 , y + 0.25, vertices_x_y) + GL.check_is_inside_tri(x + 0.25 , y - 0.25, vertices_x_y)
        media_dentro = media_dentro/4
        if(media_dentro > 0):
            newColor = [e[0]*media_dentro * (1 - GL.transp), e[1]*media_dentro * (1 - GL.transp), e[2]*media_dentro * (1 - GL.transp)]
            oldColor = gpu.GPU.read_pixel([x, y], gpu.GPU.RGB8)
            if(int(newColor[0]) != int(oldColor[0]) or int(newColor[1]) != int(oldColor[1]) or int(newColor[2]) != int(oldColor[2])):
                if GL.transp > 0:
                    newColor[0], newColor[1], newColor[2] = newColor[0] + (oldColor[0] * GL.transp), newColor[1] + (oldColor[1] * GL.transp), newColor[2] + (oldColor[2] * GL.transp)
                gpu.GPU.draw_pixel(
                    [int(x), int(y)],
                    gpu.GPU.RGB8,
                    [int(newColor[0]), int(newColor[1]), int(newColor[2])],
                )

    @staticmethod
    def triangleSet2D(vertices, colors, og_points = None):
        hasZ = False
        if len(vertices) > 6:
            vertices_x_y = [vertices[0], vertices[1], vertices[3], vertices[4], vertices[6], vertices[7]]
            hasZ = True
        else:
            vertices_x_y = [vertices[0], vertices[1], vertices[2], vertices[3], vertices[4], vertices[5]]
        if GL.color_per_vertex:
            GL.triangleSet2DColorPerVertex(vertices_x_y, colors)
        elif GL.is_texture:
            GL.triangleSet2DColorPerTexture(vertices_x_y, colors)
        else:
            emissive_colors = colors["emissiveColor"]
            e = [0, 0, 0]
            for i in range(0, len(emissive_colors)):
                e[i] = int(emissive_colors[i] * 255)

            minX = int(min([vertices_x_y[0], vertices_x_y[2], vertices_x_y[4]]))
            maxX = int(max([vertices_x_y[0], vertices_x_y[2], vertices_x_y[4]]))
            minY = int(min([vertices_x_y[1], vertices_x_y[3], vertices_x_y[5]]))
            maxY = int(max([vertices_x_y[1], vertices_x_y[3], vertices_x_y[5]]))
            if hasZ:
                GL.draw_if_has_z(minX, maxX, minY, maxY, vertices_x_y, vertices, e, og_points)
            else:
                GL.draw_if_not_z(minX, maxX, minY, maxY, vertices_x_y, e)
    

    @staticmethod
    def find_z(v1, v2, v3, x, y):
        
        pv1 = [v1[0] - x, v2[1] - y]
        pv2 = [v2[0] - x, v2[1] - y]
        pv3 = [v3[0] - x, v3[1] - y]

        va = np.linalg.norm(np.cross(pv2, pv3))/2
        vb = np.linalg.norm(np.cross(pv1, pv3))/2
        vc = np.linalg.norm(np.cross(pv1, pv2))/2

        area = va + vb + vc
        
        z = ((va*v1[2]) + vb*v2[2] + vc*v3[2]) / area
        z = (z+1)/2
        
        return z

    
    @staticmethod
    def draw_if_has_z(minX, maxX, minY, maxY, vertices_x_y, vertecies_x_y_z, e, og_points=None):
        
        for x in range(minX, maxX + 1):
            for y in range(minY, maxY + 1):
                if GL.anti_aliasing:
                    z = GL.find_z(vertecies_x_y_z[:3], vertecies_x_y_z[3:6], vertecies_x_y_z[6:], x, y)
                    if(z < GL.zbuffer[y][x]):
                        GL.zbuffer[y][x] = z 
                        GL.draw_with_anti_aliasing(x, y, vertices_x_y, e)
                else:
                    if GL.check_is_inside_tri(x, y, vertices_x_y):
                        z = GL.find_z(vertecies_x_y_z[:3], vertecies_x_y_z[3:6], vertecies_x_y_z[6:], x, y)
                        if(z < GL.zbuffer[y][x]):
                            GL.zbuffer[y][x] = z       
                            newColor = [e[0]  * (1 - GL.transp), e[1]  * (1 - GL.transp), e[2]  * (1 - GL.transp)]
                            oldColor = gpu.GPU.read_pixel([x, y], gpu.GPU.RGB8)
                            if(int(newColor[0]) != int(oldColor[0]) or int(newColor[1]) != int(oldColor[1]) or int(newColor[2]) != int(oldColor[2])):
                                if GL.transp > 0:
                                    newColor[0], newColor[1], newColor[2] = newColor[0] + (oldColor[0] * GL.transp), newColor[1] + (oldColor[1] * GL.transp), newColor[2] + (oldColor[2] * GL.transp)
                                if GL.headlight:
                                    r, g, b = GL.calc_color_with_light(int(newColor[0]), int(newColor[1]), int(newColor[2]), og_points)
                                else:
                                    r, g, b = int(newColor[0]), int(newColor[1]), int(newColor[2])
                                gpu.GPU.draw_pixel(
                                    [int(x), int(y)],
                                    gpu.GPU.RGB8,
                                    [r, g, b],
                                )
    @staticmethod
    def calc_color_with_light(og_r, og_g, og_b, vertices_x_y_z):
        
        v1 = vertices_x_y_z[:3]
        v2 = vertices_x_y_z[3:6]
        v3 = vertices_x_y_z[6:]
        
        pv1 = [v3[0] - v1[0], v3[1] - v1[1], v3[2] - v1[2]]
        pv2 = [v2[0] - v1[0], v2[1] - v1[1], v2[2] - v1[2]]

        
        norm_v1 = np.cross(pv2, pv1)

        luz = [0, 0, 1]

        res = np.dot(luz, norm_v1)
        res = 0 if res < 0 else res * 10
      
        r = og_r * res
        g = og_g * res
        b = og_b * res
        
        return r, g, b

    @staticmethod
    def draw_if_not_z(minX, maxX, minY, maxY, vertices_x_y, e):
        for x in range(minX, maxX + 1):
            for y in range(minY, maxY + 1):
                if GL.anti_aliasing:
                   GL.draw_with_anti_aliasing(x, y, vertices_x_y, e)
                else:
                    if GL.check_is_inside_tri(x, y, vertices_x_y):                    
                        newColor = [e[0]  * (1 - GL.transp), e[1]  * (1 - GL.transp), e[2]  * (1 - GL.transp)]
                        oldColor = gpu.GPU.read_pixel([x, y], gpu.GPU.RGB8)
                        if(int(newColor[0]) != int(oldColor[0]) or int(newColor[1]) != int(oldColor[1]) or int(newColor[2]) != int(oldColor[2])):
                            if GL.transp > 0:
                                newColor[0], newColor[1], newColor[2] = newColor[0] + (oldColor[0] * GL.transp), newColor[1] + (oldColor[1] * GL.transp), newColor[2] + (oldColor[2] * GL.transp)
                            
                            gpu.GPU.draw_pixel(
                                [int(x), int(y)],
                                gpu.GPU.RGB8,
                                [int(newColor[0]), int(newColor[1]), int(newColor[2])],
                            )

    @staticmethod
    def triangleSet(point, colors):
        try:
            if colors["transparency"]:
                GL.transp = colors["transparency"]
        except:
            pass
        
        n_triangle = int(len(point) / 9)

        final_matrix = np.matmul(GL.look_at, GL.transformation_matrix)
        
        for i in range(n_triangle):
            p1 = np.array(
                [[point[0 + 9 * i]], [point[1 + 9 * i]], [point[2 + 9 * i]], [1]]
            )
            p2 = np.array(
                [[point[3 + 9 * i]], [point[4 + 9 * i]], [point[5 + 9 * i]], [1]]
            )
            p3 = np.array(
                [[point[6 + 9 * i]], [point[7 + 9 * i]], [point[8 + 9 * i]], [1]]
            )
            

            p1 = np.matmul(final_matrix, p1)
            p2 = np.matmul(final_matrix, p2)
            p3 = np.matmul(final_matrix, p3)
            
            p1 = p1 / p1[3][0]
            p2 = p2 / p2[3][0]
            p3 = p3 / p3[3][0]

            og_points = [
                p1[0][0],
                p1[1][0],
                p1[2][0],
                p2[0][0],
                p2[1][0],
                p2[2][0],
                p3[0][0],
                p3[1][0],
                p3[2][0],
            ]


            p1 = np.matmul(GL.screen, p1)
            p2 = np.matmul(GL.screen, p2)
            p3 = np.matmul(GL.screen, p3)
            points = [
                int(p1[0][0]),
                int(p1[1][0]),
                p1[2][0],
                int(p2[0][0]),
                int(p2[1][0]),
                p2[2][0],
                int(p3[0][0]),
                int(p3[1][0]),
                p3[2][0],
            ]
           
            GL.triangleSet2D(points, colors, og_points)

    @staticmethod
    def calc_rotation(rot):
        qi = rot[0] * (math.sin(rot[3] / 2))
        qj = rot[1] * (math.sin(rot[3] / 2))
        qk = rot[2] * (math.sin(rot[3] / 2))
        qr = math.cos(rot[3] / 2)

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
        GL.P_mat = P
        GL.screen = np.array(
            [
                [GL.width / 2, 0, 0, GL.width / 2],
                [0, -GL.height / 2, 0, GL.height / 2],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )

        GL.look_at = np.matmul(P, GL.look_at)

        #GL.look_at = np.matmul(screen, GL.look_at)

        # print("Viewpoint : ", end="")
        # print("position = {0} ".format(position), end="")
        # print("orientation = {0} ".format(orientation), end="")
        # print("fieldOfView = {0} ".format(fieldOfView))
        # print("Look at = {0}".format(GL.look_at))

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
        #print("Transform : ", end="")
        GL.stack.append(GL.transformation_matrix)

        if translation:
            #print(
            #    "translation = {0} ".format(translation), end=""
            #)  # imprime no terminal
            translation_matrix = np.array(
                [
                    [1, 0, 0, translation[0]],
                    [0, 1, 0, translation[1]],
                    [0, 0, 1, translation[2]],
                    [0, 0, 0, 1],
                ]
            )
        if scale:
            #print("scale = {0} ".format(scale), end="")  # imprime no terminal
            scale_matrix = np.array(
                [
                    [scale[0], 0, 0, 0],
                    [0, scale[1], 0, 0],
                    [0, 0, scale[2], 0],
                    [0, 0, 0, 1],
                ]
            )
        if rotation:
            pass
            #print("rotation = {0} ".format(rotation), end="")  # imprime no terminal

        temp = np.matmul(translation_matrix, GL.calc_rotation(rotation))
        temp = np.matmul(temp, scale_matrix)
        GL.transformation_matrix = np.matmul(GL.transformation_matrix, temp)
        
    

    @staticmethod
    def transform_out():
        """Função usada para renderizar (na verdade coletar os dados) de Transform."""
        # A função transform_out será chamada quando se sair em um nó X3D do tipo Transform do
        # grafo de cena. Não são passados valores, porém quando se sai de um nó transform se
        # deverá recuperar a matriz de transformação dos modelos do mundo da estrutura de
        # pilha implementada.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        
        GL.transformation_matrix = GL.stack.pop()
       

    @staticmethod
    def triangleStripSet(point, stripCount, colors):
        pontos = [point[0], point[1], point[2], point[3], point[4], point[5], point[6], point[7], point[8]]
        
        GL.triangleSet(pontos, colors)
    
        for i in range(3, stripCount[0]):
            pontos[0] = pontos[3] 
            pontos[1] = pontos[4] 
            pontos[2] = pontos[5] 
            pontos[3] = pontos[6] 
            pontos[4] = pontos[7] 
            pontos[5] = pontos[8] 
            pontos[6] = point[(i * 3)] 
            pontos[7] = point[(i * 3) + 1] 
            pontos[8] = point[(i * 3) + 2] 
            temp = pontos.copy()
            if(i % 2 != 0):
                temp2 = [0, 0, 0]
                temp2[0] = pontos[6] 
                temp2[1] = pontos[7]
                temp2[2] = pontos[8]
                pontos[6] = pontos[3] 
                pontos[7] = pontos[4] 
                pontos[8] = pontos[5] 
                pontos[3] = temp2[0] 
                pontos[4] = temp2[1] 
                pontos[5] = temp2[2] 

            GL.triangleSet(pontos, colors)

            pontos = temp.copy()
      

    @staticmethod
    def indexedTriangleStripSet(point, index, colors):
        
        pontos ={}
        for i, e in enumerate(index):
            if e != -1:
                pontos[i] = [point[i*3], point[(i*3)+1], point[(i*3)+2]]
    
        for i in range(0, len(pontos)-2):
            if(i % 2 != 0):
                drawPontos = pontos[i] + pontos[i+2] + pontos[i+1]
            else:
                drawPontos = pontos[i] + pontos[i+1] + pontos[i+2]
            
            GL.triangleSet(drawPontos, colors)
            


    @staticmethod
    def box(size, colors):
        newSizeX = size[0]/2
        newSizeY = size[1]/2
        newSizeZ = size[2]/2

        vertices = [-1*newSizeX, 1*newSizeY, -1*newSizeZ, 
                    -1*newSizeX, 1*newSizeY, 1*newSizeZ, 
                    1*newSizeX, 1*newSizeY, 1*newSizeZ, 
                    1*newSizeX, 1*newSizeY, -1*newSizeZ, 
                    -1*newSizeX, -1*newSizeY, -1*newSizeZ, 
                    -1*newSizeX, -1*newSizeY, 1*newSizeZ, 
                    1*newSizeX, -1*newSizeY, 1*newSizeZ, 
                    1*newSizeX, -1*newSizeY, -1*newSizeZ]

        
        triangles = [[0, 1, 3, -1, 1, 2, 3, -1],
                     [0, 4, 1, -1, 4, 5, 1, -1],
                     [1, 5, 2, -1, 5, 6, 2, -1],
                     [2, 6, 3, -1, 6, 7, 3, -1],
                     [3, 7, 0, -1, 7, 4, 0, -1],
                     [4, 7, 5, -1, 7, 6, 5, -1]]
    

        
        
        for i in range(len(triangles)):
            GL.indexedFaceSet(coord=vertices, coordIndex=triangles[i], colors=colors, colorPerVertex=False, color=None, colorIndex=None, texCoord=None, texCoordIndex=None, current_texture=None)

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
        
        if current_texture:
            GL.is_texture = True
            GL.cur_text = gpu.GPU.load_texture(current_texture[0])
            pontos ={}
            textura = {}
            for i in range(int(len(coord)/3)):
                pontos[i] = [coord[i*3], coord[(i*3)+1], coord[(i*3)+2]]
            for i in range(int(len(texCoord)/2)):    
                textura[i] = [texCoord[i*2], texCoord[(i*2)+1]]
            idex = []
            for i in coordIndex:
                if(i >= 0):
                    idex.append(i)
                else:
                    drawPoints = []
                    drawText = []
                    for e in idex:
                        drawPoints += pontos[e]
                        drawText += textura[e]

                    GL.triangleStripSet(drawPoints, [int(len(idex)/3)], drawText)
                    idex = []
        elif(not color):
            pontos ={}
            for i in range(int(len(coord)/3)):
                pontos[i] = [coord[i*3], coord[(i*3)+1], coord[(i*3)+2]]
            idex = []
            for i in coordIndex:
                if(i >= 0):
                    idex.append(i)
                else:
                    drawPoints = []
                    for e in idex:
                        drawPoints += pontos[e]
                    GL.triangleStripSet(drawPoints, [int(len(idex)/3)], colors)
                    idex = []
        else:
            GL.color_per_vertex = True
            pontos ={}
            cores = {}
            for i in range(int(len(coord)/3)):
                pontos[i] = [coord[i*3], coord[(i*3)+1], coord[(i*3)+2]]
                cores[i] = [color[i*3], color[(i*3)+1], color[(i*3)+2]]
            idex = []
            for i in coordIndex:
                if(i >= 0):
                    idex.append(i)
                else:
                    drawPoints = []
                    drawColors = []
                    for e in idex:
                        drawPoints += pontos[e]
                        drawColors += cores[e]

                    GL.triangleStripSet(drawPoints, [int(len(idex)/3)], drawColors)
                    idex = []


    @staticmethod
    def sphere(radius, colors):
        points = []
        for b in np.arange(0, math.pi, math.pi/50):
            ring = []
            z = radius*math.cos(b)
            for a in np.arange(0, 2*math.pi, 2*math.pi/50):
                x = radius*math.cos(a)*math.sin(b)
                y = radius*math.sin(a)*math.sin(b)
                ring.append([x, y, z])
            points.append(ring)
        
        
        for e in range(len(points)-1):
            triangleStrip = []
            for i in range(len(points[e])):
                triangleStrip += points[e+1][i]
                triangleStrip += points[e][i]
            
            triangleStrip += points[e+1][-1]
            triangleStrip += points[e][-1]
            triangleStrip += points[e+1][0]
            triangleStrip += points[e][0]
            
            indexed_cords = [i for i in range(int(len(triangleStrip)/3))]
            GL.indexedTriangleStripSet(triangleStrip, indexed_cords, colors)
       
    @staticmethod
    def navigationInfo(headlight):
        """Características físicas do avatar do visualizador e do modelo de visualização."""
        # O campo do headlight especifica se um navegador deve acender um luz direcional que
        # sempre aponta na direção que o usuário está olhando. Definir este campo como TRUE
        # faz com que o visualizador forneça sempre uma luz do ponto de vista do usuário.
        # A luz headlight deve ser direcional, ter intensidade = 1, cor = (1 1 1),
        # ambientIntensity = 0,0 e direção = (0 0 −1).
        intensidade = 1
        cor = (1,1,1)
        ambientIntensity = 0,0
        direcao = (0, 0, -1)
        GL.headlight = True
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
