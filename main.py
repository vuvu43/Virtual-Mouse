from classHandDetec import hand_detector
import cv2 as cv
import math
import pyautogui
pyautogui.FAILSAFE=False





def main():
    cap = cv.VideoCapture(0) #captura de vídeo ao vivo da webcam
    cap.set(3, 456) #tamanho da janela no eixo x
    cap.set(4, 356) #tamanho da janela no eixo y
    detector = hand_detector(detectionCon=0.8) #instancia a classe para detectar as mãos
    tam_monitor = pyautogui.size()
    mouse = (tam_monitor[0]//2, tam_monitor[1]//2) #posicao do mouse na tela

    while True:
        _, img = cap.read() #lê da webcam
        imt = detector.find_hands(img) #deteca onde está a mão
        lmList = detector.find_position(img) #detecta landmarks e desenha se desejado

        #Dimensões da imagem capturada
        h, w, _ = img.shape
        
        #Dimensões do retângulo
        recw = tam_monitor[0] // 4
        rech = tam_monitor[1] // 4
        
        #Coordenadas do canto superior esquerdo do retângulo centralizado
        top_left_x = (w - recw) // 2
        top_left_y = (h - rech) // 2
        
        #Coordenadas do canto inferior direito do retângulo centralizado
        bottom_right_x = top_left_x + recw
        bottom_right_y = top_left_y + rech
        
        #Desenha o retângulo centralizado
        img = cv.rectangle(img, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (255,0,255), 2)

        if lmList: #se detectar algo
            levantados:int = detector.fingers_up(img, lmList) #lista de dedos levantados
            ind_x, ind_y = lmList[8][1], lmList[8][2] #posição do dedo indicador
            
            #calcula a posição relativa dentro do retângulo
            rel_x = (ind_x - top_left_x) / recw
            rel_y = (ind_y - top_left_y) / rech
                    
            #traduz a posição relativa para a posição na tela
            screen_x = int(rel_x * tam_monitor[0])
            screen_y = int(rel_y * tam_monitor[1])

            mouse = (tam_monitor[0] - screen_x, screen_y) #posicao do mouse na tela

            #se todos os dedos estiverm levantados
            if sum(levantados) == 5:
                exit(0) #termina o programa

            #se indicador e dedo do meio levantados
            elif (levantados[1] == 1) and (levantados[2] == 1): 
                #desenha na ponta do indicador
                cv.circle(img, (ind_x, ind_y), 5, (255,0,0), cv.FILLED)

                medx, medy = lmList[12][1], lmList[12][2] #ponta do dedo do meio
                cv.circle(img, (medx, medy), 5, (255,0,0), cv.FILLED)

                cv.line(img, (ind_x, ind_y), (medx, medy), (255,0,0), 2) #espaço entre os dedos

                ponto_medio = ((ind_x + medx)//2, (ind_y + medy)//2) 
                cv.circle(img, ponto_medio, 6, (255,0,0), cv.FILLED) #desenha um circulo entre os dedos
         
                if math.hypot(medx - ind_x, medy - ind_y) < 25: #TOCOU Os DEDOS

                    #sinaliza que os dedos tocaram
                    cv.circle(img, ponto_medio, 6, (0,255,0), cv.FILLED)
                    
                    #clica onde mouse está posicionado
                    pyautogui.leftClick(mouse)
                
            elif levantados[1] == 1: #se apenas indicador levantado

                #se o dedo indicador estiver dentro do retângulo
                if (top_left_x <= ind_x <= bottom_right_x) and (top_left_y <= ind_y <= bottom_right_y):

                    pyautogui.moveTo(mouse) #move o mouse para a posição calculada
                    cv.circle(img, (ind_x, ind_y), 7, (255,0,0), cv.FILLED) #desenha um círculo no indicador
                
        img = cv.flip(img, 1) #inverte a imagem         
        cv.imshow("Webcam", img)
        cv.waitKey(1)


if __name__ == "__main__":
    main()
