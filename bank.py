import pygame
import os

class Bank:
    def __init__(self, initial_amount=1000, font=None):
        self.amount = initial_amount
        self.bet = 10
        self.min_bet = 1
        self.max_bet = 500
        self.slider_rect = pygame.Rect(0, 0, 200, 10)
        self.slider_handle_rect = pygame.Rect(0, 0, 20, 20)
        self.dragging = False
        self.font = font or pygame.font.SysFont("serif", 20, bold=True)

        # Carregar as imagens da pasta "money" e redimensionar
        self.img_dinheiro = pygame.image.load(os.path.join(os.path.dirname(__file__), "money", "dinheiro.png")).convert_alpha()
        self.img_dinheiro = pygame.transform.scale(self.img_dinheiro, (self.img_dinheiro.get_width() // 2, self.img_dinheiro.get_height() // 2))
        self.img_carteira = pygame.image.load(os.path.join(os.path.dirname(__file__), "money", "carteira.png")).convert_alpha()
        self.img_carteira = pygame.transform.scale(self.img_carteira, (self.img_carteira.get_width() // 2, self.img_carteira.get_height() // 2))

    def draw(self, screen, position):
        x, y = position

        # Mostrar imagem da carteira ao lado do valor do banco (estática)
        screen.blit(self.img_carteira, (x - 40, y - 8))

        # Mostrar saldo do banco
        amt_text = self.font.render(f"Banco: ${self.amount}", True, (255, 255, 255))
        screen.blit(amt_text, (x, y))

        # Barra do slider
        self.slider_rect.topleft = (x, y + 30)
        pygame.draw.rect(screen, (180, 180, 180), self.slider_rect)

        # Posição do botão do slider
        handle_x = x + int((self.bet - self.min_bet) / (self.max_bet - self.min_bet) * (self.slider_rect.width - self.slider_handle_rect.width))
        handle_y = y + 25
        self.slider_handle_rect.topleft = (handle_x, handle_y)
        pygame.draw.rect(screen, (255, 255, 0), self.slider_handle_rect)

        # Mostrar aposta atual (sem imagem do dinheiro aqui, pois ela anima)
        bet_text = self.font.render(f"Aposta: ${self.bet}", True, (255, 255, 255))
        screen.blit(bet_text, (x, y + 60))

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.slider_handle_rect.collidepoint(event.pos):
                self.dragging = True
        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            rel_x = event.pos[0] - self.slider_rect.x
            rel_x = max(0, min(rel_x, self.slider_rect.width - self.slider_handle_rect.width))
            self.bet = self.min_bet + int(rel_x / (self.slider_rect.width - self.slider_handle_rect.width) * (self.max_bet - self.min_bet))

    def place_bet(self):
        if self.bet > self.amount:
            return False
        self.amount -= self.bet
        return True

    def payout(self, multiplier=2):
        self.amount += int(self.bet * multiplier)
