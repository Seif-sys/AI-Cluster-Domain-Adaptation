import matplotlib.pyplot as plt

angles = [0, 15, 30, 45, 60]

# Fill these in from your results_summary.txt files
source_only = [0.9944, None, None, 0.6552, None]  # target acc before adaptation
coral       = [0.9944, None, None, 0.7031, None]  # target acc after adaptation (best lambda)
upper_bound = [None,   None, None, 0.9919, None]  # from upper bound run

plt.figure(figsize=(7, 4))
plt.plot(angles, source_only, 'o--', label='Source-only baseline')
plt.plot(angles, coral,       's-',  label='Deep CORAL (λ=25)')
plt.plot(angles, upper_bound, '^:',  label='Supervised upper bound', color='green')
plt.xlabel('Target rotation angle (degrees)')
plt.ylabel('Target test accuracy')
plt.title('CORAL adaptation vs. domain gap (Rotated MNIST)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('gap_curve.pdf')
plt.show()