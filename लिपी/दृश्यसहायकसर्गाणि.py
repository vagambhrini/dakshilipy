#!/usr/bin/env python
# coding: utf-8
"""
दृश्यसहायकसर्गाणि - Visualization Helpers for दाक्षिलिपी

This module provides visualization functions for mathematical algorithms
using Plotly, Matplotlib, and ipywidgets with Sanskrit terminology.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Rectangle, Polygon, FancyArrowPatch
from matplotlib.collections import PatchCollection
import matplotlib.patches as mpatches
from IPython.display import HTML, display
import warnings

# Plotly imports with fallback
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    warnings.warn("Plotly not available. Some visualizations will be disabled.")

# ipywidgets imports with fallback
try:
    from ipywidgets import interact, interactive, IntSlider, FloatSlider, Dropdown
    import ipywidgets as widgets
    WIDGETS_AVAILABLE = True
except ImportError:
    WIDGETS_AVAILABLE = False
    warnings.warn("ipywidgets not available. Interactive controls will be disabled.")

# K3D imports with fallback
try:
    import k3d
    K3D_AVAILABLE = True
except ImportError:
    K3D_AVAILABLE = False
    warnings.warn("K3D not available. 3D geometry visualizations will use Plotly instead.")


# =============================================================================
# मेरु प्रस्तार (Pascal's Triangle) Visualizations
# =============================================================================

def मेरुसारिणी_दर्शय(पंक्ति_संख्या, शैली='त्रिभुज'):
    """
    मेरु प्रस्तार (Pascal's Triangle) को दृश्य रूप में दर्शाता है।

    Parameters:
    -----------
    पंक्ति_संख्या : int
        मेरु की पंक्तियों की संख्या
    शैली : str
        'त्रिभुज' (triangular), 'सारिणी' (table), or 'ताप' (heatmap)
    """
    if not PLOTLY_AVAILABLE:
        return _मेरु_matplotlib(पंक्ति_संख्या)

    # Generate Pascal's triangle
    मेरु = _मेरु_गणना(पंक्ति_संख्या)

    if शैली == 'त्रिभुज':
        return _मेरु_त्रिभुज_plotly(मेरु, पंक्ति_संख्या)
    elif शैली == 'सारिणी':
        return _मेरु_सारिणी_plotly(मेरु, पंक्ति_संख्या)
    elif शैली == 'ताप':
        return _मेरु_ताप_plotly(मेरु, पंक्ति_संख्या)
    else:
        return _मेरु_त्रिभुज_plotly(मेरु, पंक्ति_संख्या)


def _मेरु_गणना(n):
    """मेरु प्रस्तार (Pascal's Triangle) की गणना करता है।"""
    मेरु = [[1]]
    for i in range(1, n):
        पूर्व = मेरु[-1]
        नवीन = [1]
        for j in range(len(पूर्व) - 1):
            नवीन.append(पूर्व[j] + पूर्व[j + 1])
        नवीन.append(1)
        मेरु.append(नवीन)
    return मेरु


def _मेरु_त्रिभुज_plotly(मेरु, n):
    """Plotly के साथ त्रिभुजाकार मेरु दृश्य।"""
    fig = go.Figure()

    # Add text annotations for each number
    for i, पंक्ति in enumerate(मेरु):
        for j, मान in enumerate(पंक्ति):
            x = j - i/2
            y = n - i - 1
            fig.add_annotation(
                x=x, y=y,
                text=str(मान),
                showarrow=False,
                font=dict(size=max(8, 16 - n//2), color='darkblue'),
                bgcolor='lightyellow',
                bordercolor='orange',
                borderwidth=1,
                borderpad=4,
            )

    fig.update_layout(
        title=dict(
            text=f'मेरु प्रस्तार ({n} पंक्तियाँ)',
            font=dict(size=20)
        ),
        xaxis=dict(visible=False, range=[-n/2-1, n/2+1]),
        yaxis=dict(visible=False, range=[-1, n+1]),
        width=max(400, n * 60),
        height=max(300, n * 50),
        plot_bgcolor='white',
    )

    return fig


def _मेरु_ताप_plotly(मेरु, n):
    """Plotly heatmap के साथ मेरु दृश्य।"""
    # Create a matrix for heatmap
    आव्यूह = np.zeros((n, 2*n - 1))
    आव्यूह[:] = np.nan

    for i, पंक्ति in enumerate(मेरु):
        offset = n - 1 - i
        for j, मान in enumerate(पंक्ति):
            आव्यूह[i, offset + 2*j] = मान

    fig = go.Figure(data=go.Heatmap(
        z=आव्यूह,
        colorscale='YlOrRd',
        showscale=True,
        hoverongaps=False,
    ))

    fig.update_layout(
        title='मेरु प्रस्तार - ताप मानचित्र',
        xaxis=dict(visible=False),
        yaxis=dict(autorange='reversed'),
    )

    return fig


def _मेरु_सारिणी_plotly(मेरु, n):
    """Plotly table के साथ मेरु दृश्य।"""
    # Create table representation
    max_len = len(मेरु[-1])
    headers = [f'स्थान {i}' for i in range(max_len)]

    cells = []
    for पंक्ति in मेरु:
        पूरक = [''] * (max_len - len(पंक्ति))
        cells.append(list(पंक्ति) + पूरक)

    fig = go.Figure(data=[go.Table(
        header=dict(values=headers, fill_color='paleturquoise', align='center'),
        cells=dict(values=list(zip(*cells)), fill_color='lavender', align='center')
    )])

    fig.update_layout(title='मेरु प्रस्तार - सारिणी रूप')

    return fig


def _मेरु_matplotlib(n):
    """Matplotlib fallback for मेरु visualization."""
    मेरु = _मेरु_गणना(n)

    fig, ax = plt.subplots(figsize=(max(6, n), max(4, n*0.7)))
    ax.set_aspect('equal')

    for i, पंक्ति in enumerate(मेरु):
        for j, मान in enumerate(पंक्ति):
            x = j - i/2
            y = n - i - 1
            ax.text(x, y, str(मान), ha='center', va='center',
                   fontsize=max(6, 14 - n//3),
                   bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='orange'))

    ax.set_xlim(-n/2 - 1, n/2 + 1)
    ax.set_ylim(-1, n + 1)
    ax.axis('off')
    ax.set_title(f'मेरु प्रस्तार ({n} पंक्तियाँ)', fontsize=14)

    return fig


def मेरु_चलचित्र(पंक्ति_संख्या, अवधि=500):
    """
    मेरु प्रस्तार का चलचित्र (animation) बनाता है।

    Parameters:
    -----------
    पंक्ति_संख्या : int
        अन्तिम पंक्ति संख्या
    अवधि : int
        प्रति फ्रेम मिलीसेकण्ड
    """
    मेरु = _मेरु_गणना(पंक्ति_संख्या)

    fig, ax = plt.subplots(figsize=(max(8, पंक्ति_संख्या), max(6, पंक्ति_संख्या*0.7)))
    ax.set_xlim(-पंक्ति_संख्या/2 - 1, पंक्ति_संख्या/2 + 1)
    ax.set_ylim(-1, पंक्ति_संख्या + 1)
    ax.axis('off')
    ax.set_title('मेरु प्रस्तार - निर्माण', fontsize=14)

    texts = []

    def init():
        return texts

    def update(frame):
        # Clear previous texts
        for txt in texts:
            txt.remove()
        texts.clear()

        # Draw up to current frame
        for i in range(min(frame + 1, len(मेरु))):
            पंक्ति = मेरु[i]
            for j, मान in enumerate(पंक्ति):
                x = j - i/2
                y = पंक्ति_संख्या - i - 1
                color = 'lightgreen' if i == frame else 'lightyellow'
                txt = ax.text(x, y, str(मान), ha='center', va='center',
                            fontsize=max(6, 14 - पंक्ति_संख्या//3),
                            bbox=dict(boxstyle='round', facecolor=color, edgecolor='orange'))
                texts.append(txt)

        return texts

    anim = FuncAnimation(fig, update, frames=पंक्ति_संख्या,
                        init_func=init, interval=अवधि, blit=False)
    plt.close(fig)
    return HTML(anim.to_jshtml())


# =============================================================================
# Cellular Automata Connection (Rule 60)
# =============================================================================

def कोशिका_स्वचालित_दर्शय(पंक्ति_संख्या, नियम=60):
    """
    कोशिका स्वचालित (Cellular Automata) - मेरु से सम्बन्ध दर्शाता है।

    Rule 60: XOR rule that generates Sierpinski triangle pattern
    """
    चौड़ाई = 2 * पंक्ति_संख्या + 1
    जाल = np.zeros((पंक्ति_संख्या, चौड़ाई), dtype=int)
    जाल[0, पंक्ति_संख्या] = 1  # Initial state: single cell

    # Generate using rule
    for i in range(1, पंक्ति_संख्या):
        for j in range(1, चौड़ाई - 1):
            बायाँ = जाल[i-1, j-1]
            मध्य = जाल[i-1, j]
            दायाँ = जाल[i-1, j+1]
            पैटर्न = बायाँ * 4 + मध्य * 2 + दायाँ
            जाल[i, j] = (नियम >> पैटर्न) & 1

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(जाल, cmap='binary', interpolation='nearest')
    ax.set_title(f'कोशिका स्वचालित - नियम {नियम}\n(मेरु प्रस्तार mod 2 से सम्बन्धित)', fontsize=12)
    ax.axis('off')

    return fig


# =============================================================================
# द्विपद वितरण (Binomial Distribution) Visualization
# =============================================================================

def द्विपद_वितरण_दर्शय(n, k_highlight=None):
    """
    द्विपद गुणांक C(n,k) का वितरण दर्शाता है।
    लघु-क्रिया प्रत्यय से सम्बन्धित।
    """
    from scipy.special import comb

    k_values = list(range(n + 1))
    coefficients = [int(comb(n, k, exact=True)) for k in k_values]

    if PLOTLY_AVAILABLE:
        colors = ['lightblue'] * len(k_values)
        if k_highlight is not None and 0 <= k_highlight <= n:
            colors[k_highlight] = 'orange'

        fig = go.Figure(data=[go.Bar(
            x=[f'C({n},{k})' for k in k_values],
            y=coefficients,
            marker_color=colors,
            text=coefficients,
            textposition='outside'
        )])

        fig.update_layout(
            title=f'द्विपद गुणांक वितरण: n = {n}<br>लघु-क्रिया प्रत्यय - k लघु अक्षरों वाले छन्दों की संख्या',
            xaxis_title='द्विपद गुणांक',
            yaxis_title='संख्या',
            showlegend=False
        )
        return fig
    else:
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['lightblue' if k != k_highlight else 'orange' for k in k_values]
        ax.bar(k_values, coefficients, color=colors)
        ax.set_xlabel('k (लघु अक्षरों की संख्या)')
        ax.set_ylabel('C(n,k)')
        ax.set_title(f'द्विपद गुणांक वितरण: n = {n}')
        for i, v in enumerate(coefficients):
            ax.text(i, v + 0.1, str(v), ha='center')
        return fig


# =============================================================================
# Binary Conversion Visualization (for नष्ट/उद्दिष्ट)
# =============================================================================

def द्विआधारी_रूपान्तरण_दर्शय(संख्या, स्थान):
    """
    नष्ट प्रत्यय के लिए द्विआधारी रूपान्तरण का चरण-दर-चरण दृश्य।

    Parameters:
    -----------
    संख्या : int
        Index number to convert
    स्थान : int
        Number of binary digits (syllables)
    """
    # Get binary representation
    द्विआधारी = bin(संख्या)[2:].zfill(स्थान)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left: Division steps
    ax1 = axes[0]
    ax1.set_title('भाग विधि (Division Method)', fontsize=12)
    ax1.axis('off')

    शेष_सूची = []
    मान = संख्या
    y = 0.9
    for i in range(स्थान):
        भागफल = मान // 2
        शेष = मान % 2
        शेष_सूची.append(शेष)
        ax1.text(0.1, y, f'{मान} ÷ 2 = {भागफल} शेष {शेष}', fontsize=11,
                transform=ax1.transAxes)
        मान = भागफल
        y -= 0.12

    # Right: Pattern representation
    ax2 = axes[1]
    ax2.set_title('छन्द प्रतिरूप (Metrical Pattern)', fontsize=12)
    ax2.axis('off')

    for i, अंक in enumerate(द्विआधारी):
        प्रतीक = 'ग' if अंक == '0' else 'ल'
        रंग = 'lightgreen' if अंक == '0' else 'lightcoral'
        ax2.add_patch(Rectangle((i * 0.15 + 0.1, 0.4), 0.12, 0.2,
                                facecolor=रंग, edgecolor='black'))
        ax2.text(i * 0.15 + 0.16, 0.5, प्रतीक, fontsize=14, ha='center', va='center')
        ax2.text(i * 0.15 + 0.16, 0.3, अंक, fontsize=10, ha='center', va='top')

    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.text(0.5, 0.75, f'क्रमांक {संख्या} → द्विआधारी: {द्विआधारी}',
            fontsize=12, ha='center', transform=ax2.transAxes)

    plt.tight_layout()
    return fig


# =============================================================================
# Geometric Constructions (शुल्बसूत्र)
# =============================================================================

def पाइथागोरस_दर्शय(a=3, b=4):
    """
    बौधायन/पाइथागोरस प्रमेय का दृश्य प्रदर्शन।
    """
    c = np.sqrt(a**2 + b**2)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_aspect('equal')

    # Draw the right triangle
    त्रिभुज = Polygon([(0, 0), (a, 0), (0, b)], fill=True,
                      facecolor='lightblue', edgecolor='blue', linewidth=2)
    ax.add_patch(त्रिभुज)

    # Draw squares on each side
    # Square on side a
    ax.add_patch(Rectangle((0, -a), a, a, fill=True,
                           facecolor='lightyellow', edgecolor='orange', linewidth=2, alpha=0.7))
    ax.text(a/2, -a/2, f'a² = {a**2}', ha='center', va='center', fontsize=12)

    # Square on side b
    ax.add_patch(Rectangle((-b, 0), b, b, fill=True,
                           facecolor='lightgreen', edgecolor='green', linewidth=2, alpha=0.7))
    ax.text(-b/2, b/2, f'b² = {b**2}', ha='center', va='center', fontsize=12)

    # Square on hypotenuse (rotated)
    angle = np.arctan2(b, a)
    # Calculate corners of the rotated square
    corners = [
        (0, 0),
        (c * np.cos(angle - np.pi/2), c * np.sin(angle - np.pi/2)),
        (a + c * np.cos(angle - np.pi/2), b + c * np.sin(angle - np.pi/2)),
        (a, b)
    ]
    ax.add_patch(Polygon(corners, fill=True,
                        facecolor='lightcoral', edgecolor='red', linewidth=2, alpha=0.7))
    ax.text(a/2 + 1, b/2 + 1, f'c² = {c**2:.2f}', ha='center', va='center', fontsize=12)

    # Labels
    ax.text(a/2, -0.3, f'a = {a}', ha='center', fontsize=10)
    ax.text(-0.3, b/2, f'b = {b}', ha='center', fontsize=10, rotation=90)
    ax.text(a/2 + 0.3, b/2 + 0.3, f'c = {c:.2f}', ha='center', fontsize=10)

    ax.set_xlim(-b - 1, a + c + 1)
    ax.set_ylim(-a - 1, b + c + 1)
    ax.set_title('बौधायन प्रमेय\n"दीर्घचतुरश्रस्याक्ष्णया रज्जुः..."', fontsize=14)
    ax.axis('off')

    return fig


def वर्गमूल_दो_सन्निकटन_दर्शय(पद_संख्या=10):
    """
    बौधायन का √2 सन्निकटन - अभिसरण दृश्य।
    √2 ≈ 1 + 1/3 + 1/(3×4) - 1/(3×4×34)
    """
    # Different approximations over history
    सन्निकटन = [
        (1, "1"),
        (1 + 1/3, "1 + 1/3"),
        (1 + 1/3 + 1/12, "1 + 1/3 + 1/12"),
        (1 + 1/3 + 1/12 - 1/408, "बौधायन: 1 + 1/3 + 1/12 - 1/408"),
        (577/408, "577/408"),
    ]

    सही_मान = np.sqrt(2)

    if PLOTLY_AVAILABLE:
        fig = go.Figure()

        x = list(range(len(सन्निकटन)))
        y = [s[0] for s in सन्निकटन]
        labels = [s[1] for s in सन्निकटन]

        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode='lines+markers',
            name='सन्निकटन',
            text=labels,
            hoverinfo='text+y'
        ))

        fig.add_hline(y=सही_मान, line_dash="dash", line_color="red",
                     annotation_text=f"√2 = {सही_मान:.10f}")

        fig.update_layout(
            title='√2 का बौधायन सन्निकटन - अभिसरण',
            xaxis_title='चरण',
            yaxis_title='मान',
        )
        return fig
    else:
        fig, ax = plt.subplots(figsize=(10, 6))
        x = list(range(len(सन्निकटन)))
        y = [s[0] for s in सन्निकटन]
        ax.plot(x, y, 'bo-', markersize=10)
        ax.axhline(y=सही_मान, color='r', linestyle='--', label=f'√2 = {सही_मान:.10f}')
        ax.set_xlabel('चरण')
        ax.set_ylabel('मान')
        ax.set_title('√2 का बौधायन सन्निकटन')
        ax.legend()
        return fig


# =============================================================================
# Astronomical Visualizations (ज्योतिष)
# =============================================================================

def ग्रह_कक्षा_दर्शय_2d(ग्रह_सूची=None):
    """
    ग्रहों की कक्षाओं का 2D दृश्य।
    """
    if ग्रह_सूची is None:
        ग्रह_सूची = [
            {'नाम': 'बुध', 'दूरी': 0.39, 'वर्ण': 'gray'},
            {'नाम': 'शुक्र', 'दूरी': 0.72, 'वर्ण': 'yellow'},
            {'नाम': 'पृथ्वी', 'दूरी': 1.0, 'वर्ण': 'blue'},
            {'नाम': 'मंगल', 'दूरी': 1.52, 'वर्ण': 'red'},
            {'नाम': 'गुरु', 'दूरी': 5.2, 'वर्ण': 'orange'},
        ]

    if PLOTLY_AVAILABLE:
        fig = go.Figure()

        # Sun at center
        fig.add_trace(go.Scatter(
            x=[0], y=[0],
            mode='markers',
            marker=dict(size=30, color='gold'),
            name='सूर्य'
        ))

        # Planet orbits and positions
        for ग्रह in ग्रह_सूची:
            θ = np.linspace(0, 2*np.pi, 100)
            x = ग्रह['दूरी'] * np.cos(θ)
            y = ग्रह['दूरी'] * np.sin(θ)

            fig.add_trace(go.Scatter(
                x=x, y=y,
                mode='lines',
                line=dict(dash='dash', color='lightgray'),
                showlegend=False
            ))

            # Planet position (random angle for demo)
            कोण = np.random.uniform(0, 2*np.pi)
            fig.add_trace(go.Scatter(
                x=[ग्रह['दूरी'] * np.cos(कोण)],
                y=[ग्रह['दूरी'] * np.sin(कोण)],
                mode='markers+text',
                marker=dict(size=15, color=ग्रह['वर्ण']),
                text=[ग्रह['नाम']],
                textposition='top center',
                name=ग्रह['नाम']
            ))

        fig.update_layout(
            title='ग्रह कक्षाएँ - सौर मण्डल',
            xaxis=dict(scaleanchor='y', scaleratio=1),
            yaxis=dict(scaleanchor='x'),
            showlegend=True
        )
        return fig
    else:
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_aspect('equal')

        ax.plot(0, 0, 'yo', markersize=20, label='सूर्य')

        for ग्रह in ग्रह_सूची:
            θ = np.linspace(0, 2*np.pi, 100)
            ax.plot(ग्रह['दूरी'] * np.cos(θ), ग्रह['दूरी'] * np.sin(θ),
                   '--', color='lightgray')
            कोण = np.random.uniform(0, 2*np.pi)
            ax.plot(ग्रह['दूरी'] * np.cos(कोण), ग्रह['दूरी'] * np.sin(कोण),
                   'o', color=ग्रह['वर्ण'], markersize=10, label=ग्रह['नाम'])

        ax.legend()
        ax.set_title('ग्रह कक्षाएँ')
        return fig


def चन्द्र_कला_दर्शय(तिथि=1):
    """
    चन्द्र कला (Moon Phase) का दृश्य।
    तिथि: 1 (प्रतिपदा) से 30 (अमावस्या)
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect('equal')

    # Full moon circle
    चन्द्र = Circle((0, 0), 1, fill=True, facecolor='lightyellow', edgecolor='gray')
    ax.add_patch(चन्द्र)

    # Shadow based on tithi
    if तिथि <= 15:  # Shukla paksha
        कला = 1 - (तिथि - 1) / 14
    else:  # Krishna paksha
        कला = (तिथि - 15) / 15

    # Create shadow ellipse
    if कला > 0:
        shadow_width = 2 * कला
        छाया = mpatches.Ellipse((0, 0), shadow_width, 2,
                                 facecolor='darkgray', edgecolor='none')
        ax.add_patch(छाया)

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)

    पक्ष = 'शुक्ल पक्ष' if तिथि <= 15 else 'कृष्ण पक्ष'
    ax.set_title(f'चन्द्र कला - तिथि {तिथि} ({पक्ष})', fontsize=14)
    ax.axis('off')

    return fig


# =============================================================================
# Interactive Widgets (if available)
# =============================================================================

def अन्तःक्रियात्मक_मेरु():
    """ipywidgets के साथ अन्तःक्रियात्मक मेरु दृश्य।"""
    if not WIDGETS_AVAILABLE:
        print("ipywidgets उपलब्ध नहीं है। कृपया `pip install ipywidgets` से स्थापित करें।")
        return None

    @interact(पंक्ति_संख्या=IntSlider(min=1, max=15, value=5, description='पंक्तियाँ:'),
              शैली=Dropdown(options=['त्रिभुज', 'ताप', 'सारिणी'], value='त्रिभुज', description='शैली:'))
    def _update(पंक्ति_संख्या, शैली):
        fig = मेरुसारिणी_दर्शय(पंक्ति_संख्या, शैली)
        if PLOTLY_AVAILABLE:
            fig.show()
        else:
            plt.show()

    return _update


def अन्तःक्रियात्मक_द्विपद():
    """ipywidgets के साथ अन्तःक्रियात्मक द्विपद वितरण।"""
    if not WIDGETS_AVAILABLE:
        print("ipywidgets उपलब्ध नहीं है।")
        return None

    @interact(n=IntSlider(min=1, max=20, value=5, description='n:'),
              k=IntSlider(min=0, max=20, value=2, description='k (हाइलाइट):'))
    def _update(n, k):
        k = min(k, n)
        fig = द्विपद_वितरण_दर्शय(n, k)
        if PLOTLY_AVAILABLE:
            fig.show()
        else:
            plt.show()

    return _update


# =============================================================================
# Utility Functions
# =============================================================================

def दर्शय_चित्र(fig):
    """Display figure appropriately based on type."""
    if PLOTLY_AVAILABLE and hasattr(fig, 'show'):
        fig.show()
    else:
        plt.show()


def सभी_दृश्य_उपलब्धता():
    """Check and report availability of visualization libraries."""
    print("दृश्य पुस्तकालय उपलब्धता:")
    print(f"  Plotly: {'✓' if PLOTLY_AVAILABLE else '✗'}")
    print(f"  ipywidgets: {'✓' if WIDGETS_AVAILABLE else '✗'}")
    print(f"  K3D: {'✓' if K3D_AVAILABLE else '✗'}")
    print(f"  Matplotlib: ✓ (always available)")
