import cv2 as cv
import mediapipe as mp
import math

class hand_detector():
    def __init__(self, mode=False, maxHands=2, modelComplexity=1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode #True se for imagem estática
        self.maxHands = maxHands #Quantidade máxima de mãos detectadas
        self.detectionCon = detectionCon #Threshold de confiança
        self.trackCon = trackCon #Threshold de confiança
        self.modelComplexity = modelComplexity #0 para básico, 1 para intermediário, 2 para avançado

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, 
                                        self.maxHands, 
                                        self.modelComplexity, 
                                        self.detectionCon, 
                                        self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils


    def find_hands(self, img, draw=True):
        """
        Método para encontrar as mãos em uma dada imagem, se draw=True, desenha na imagem os landmarks

        Entrada:
            - img: imagem provida pelo usuário como uma matriz
            - draw: True se for desejável desenhar as landmarks
        
        Saída:
            - img: imagem (Matlike) com as landmarks desenhadas
        """

        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB) #converte os canais de cor para RGB
        self.result = self.hands.process(imgRGB) #faz tracking da mão

        if self.result.multi_hand_landmarks:
            for handLms in self.result.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        
        return img


    def find_position(self, img, hand_no=0, draw_lms=list[int]) -> list[tuple[int]]:
        """
        Método para achar posição das landmarks na imagem

        Entrada:
            - img: imagem provida pelo usuário (MatLike)
            - hand_no: qual mão será desenhada os landmarks 0 para a primeira, 1 para a segunda
            - draw_lms: lista com quais landmarks devem ser exaltadas
        
        Saída:
            - lmList: lista com informações sobre a posição de um landmark
        """
        lmList = []

        if self.result.multi_hand_landmarks:
            my_hand = self.result.multi_hand_landmarks[hand_no]

            for (tag, lm) in enumerate(my_hand.landmark):
                h, w, _ = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h) #calcula em que posição da imagem está a landmark
                lmList.append((tag, cx, cy))

                if tag in draw_lms:
                    #desenha um circulo nas landmarks indicadas
                    cv.circle(img, (cx,cy), 7, (255,10,10), cv.FILLED)
        
        return lmList

    def fingers_up(self, img, lmList) -> int:
        """
        Método que dirá a quantidade de dedos levantados em uma imagem baseando-se
        na distância da ponta dos dedos até a base da mão.
        Entrada:
            - img: imagem para ser detectada

        Saída:
            - levantados: dedos que estão levantados (lista)
        """
        levantados = []
        
        fingers_id = [8, 12, 16, 20]
        
        #calculando especificamente para o dedão 
        if lmList[4][1] > lmList[2][1]:
            levantados.append(1)
        else: levantados.append(0)

        #calculando se dedos estão abertos
        for tag in fingers_id:
            if lmList[tag][2] < lmList[tag - 2][2]:
                levantados.append(1)
            else: levantados.append(0)


        return levantados
    
