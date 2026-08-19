"""
Script para testar o agente DQN treinado jogando 2048.

Uso:
    python ai/play.py [--model checkpoints/best.pt] [--games 100] [--visual]
"""

import argparse
import os
import sys
import time
import numpy as np

from game import Game2048
from dqn_agent import DQNAgent, encode_state


DIRECTION_NAMES = ['↑ UP', '→ RIGHT', '↓ DOWN', '← LEFT']


def play_visual(agent, delay=0.3):
    """Joga uma partida mostrando cada jogada no terminal."""
    game = Game2048()
    state = game.reset()
    move_count = 0

    while not game.over:
        os.system('clear' if os.name != 'nt' else 'cls')
        print(f"🎮 2048 - AI (DQN) | Jogada #{move_count}")
        print(f"{'─' * 30}")
        print()

        # Renderiza grid com cores
        for row in game.grid:
            line = ""
            for val in row:
                if val == 0:
                    line += "    ·  "
                else:
                    line += f"{val:5d}  "
            print(line)

        print()
        print(f"Score: {game.score} | Max: {game.max_tile()}")

        valid_moves = game.get_valid_moves()
        if not valid_moves:
            break

        encoded = encode_state(state)
        action = agent.select_action(encoded, valid_moves, training=False)
        print(f"Ação: {DIRECTION_NAMES[action]}")

        reward, moved = game.move(action)
        state = game.get_state()
        move_count += 1

        time.sleep(delay)

    os.system('clear' if os.name != 'nt' else 'cls')
    print(f"{'=' * 30}")
    print(f"  GAME OVER!")
    print(f"  Score final: {game.score}")
    print(f"  Max tile: {game.max_tile()}")
    print(f"  Jogadas: {move_count}")
    if game.won:
        print(f"  🏆 2048 ALCANÇADO!")
    print(f"{'=' * 30}")

    return game.score, game.max_tile(), game.won


def play_batch(agent, num_games=100):
    """Joga múltiplas partidas e mostra estatísticas."""
    scores = []
    max_tiles = []
    wins = 0

    for i in range(num_games):
        game = Game2048()
        state = game.reset()

        while not game.over:
            valid_moves = game.get_valid_moves()
            if not valid_moves:
                break
            encoded = encode_state(state)
            action = agent.select_action(encoded, valid_moves, training=False)
            game.move(action)
            state = game.get_state()

        scores.append(game.score)
        max_tiles.append(game.max_tile())
        if game.won:
            wins += 1

        if (i + 1) % 10 == 0:
            print(f"  Jogo {i+1}/{num_games} concluído | Score: {game.score} | Max: {game.max_tile()}")

    # Estatísticas
    tile_dist = {}
    for t in max_tiles:
        tile_dist[t] = tile_dist.get(t, 0) + 1

    print(f"\n{'=' * 50}")
    print(f"  RESULTADOS - {num_games} jogos")
    print(f"{'=' * 50}")
    print(f"  Score médio:    {np.mean(scores):.0f}")
    print(f"  Score mediano:  {np.median(scores):.0f}")
    print(f"  Score máximo:   {max(scores)}")
    print(f"  Score mínimo:   {min(scores)}")
    print()
    print(f"  Distribuição de max tiles:")
    for tile in sorted(tile_dist.keys()):
        pct = tile_dist[tile] / num_games * 100
        bar = '█' * int(pct / 2)
        print(f"    {tile:>5d}: {tile_dist[tile]:>3d} ({pct:5.1f}%) {bar}")
    print()
    print(f"  2048 alcançado: {wins}/{num_games} ({wins/num_games*100:.1f}%)")
    print(f"{'=' * 50}")


def main():
    parser = argparse.ArgumentParser(description='Testar agente DQN no 2048')
    parser.add_argument('--model', type=str, default='checkpoints/best.pt',
                        help='Caminho do modelo')
    parser.add_argument('--games', type=int, default=100,
                        help='Número de jogos no modo batch')
    parser.add_argument('--visual', action='store_true',
                        help='Modo visual (1 jogo animado)')
    args = parser.parse_args()

    model_path = os.path.join(os.path.dirname(__file__), args.model)
    if not os.path.exists(model_path):
        print(f"Modelo não encontrado: {model_path}")
        print("Execute primeiro: python ai/train.py")
        sys.exit(1)

    agent = DQNAgent()
    agent.load(model_path)

    if args.visual:
        play_visual(agent)
    else:
        play_batch(agent, args.games)


if __name__ == '__main__':
    main()
