import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from typing import Dict, Tuple, Callable

class WordEmbeddingsAnalyzer:
    """A class to test word embeddings models sustainability on different transformations"""
    
    def __init__(self, vectorizing_function: Callable[[str], np.ndarray]) -> None:
        """
        Args:
            vectorizing_function: A word vectorizing function
        """
        self.vectorize = vectorizing_function

    def get_stats(
        self,
        original_word: str,
        transformation_type: str,
        n_transformations: int,
        num_tests: int = 100
    ) -> Tuple[float, float]:
        """Calculate mean and normalized variance of cosine distances between original and transformed words.
        
        Applies a transformation function to a word multiple times, computes cosine distance
        between each transformed word's vector and the original vector, then returns statistics.
        
        Args:
            original_word: The original word to transform
            transformation_type: Type of transformation ('remove', 'insert', or 'shuffle')
            n_transformations: Number of transformations to apply
            num_tests: Number of tests to perform (default: 100)
            
        Returns:
            Tuple containing:
                - mean cosine distance
                - normalized variance of cosine distances
        """
        original_vector = self.vectorize(original_word)
        transformation_functions = {
            'remove': self._remove_random_symbols,
            'insert': self._insert_random_symbols,
            'shuffle': self._swap_n_pairs
        }
        transform_func = transformation_functions[transformation_type]
        distances = np.ones(num_tests)
    
        for i in range(num_tests):
            transformed_word = transform_func(original_word, n=n_transformations)
            transformed_vector = self.vectorize(transformed_word)
            distances[i] = self._cosine_distance(original_vector, transformed_vector)

        mean_distance = np.mean(distances)
        normalized_variance = np.var(distances) / (mean_distance ** 2)
        return mean_distance, normalized_variance

    def display_stats(
        self,
        word: str,
        transformation_type: str,
        n_transformations: int,
        num_tests: int = 100
    ) -> None:
        """Display statistics about word transformations' cosine distances.
        
        Computes and prints the mean and variance of cosine distances between
        the original word vector and vectors of transformed words.
        
        Args:
            word: The original word to transform
            vector: The original word's vector representation
            transformation_type: Type of transformation ('remove', 'insert', or 'shuffle')
            n_transformations: Number of transformations to apply
            num_tests: Number of tests to perform (default: 100)
        """
        mean, variance = self.get_stats(
            word, transformation_type, n_transformations, num_tests
        )
        print(f'Mean distance: {mean:.4f}, Normalized variance: {variance:.4f}')

    @staticmethod
    def _remove_random_symbols(word: str, n: int) -> str:
        """Remove n random symbols from the word.
        
        Args:
            word: Word to transform
            n: Number of chacaters to remove
            
        Returns:
            Transformed word
        """
        symbols = np.array(list(word))
        indices_to_remove = np.random.choice(symbols.shape[0], size=n, replace=False)
        mask = np.ones(symbols.shape[0], dtype=bool)
        mask[indices_to_remove] = False
        return ''.join(symbols[mask])

    @staticmethod
    def _insert_random_symbols(word: str, n: int) -> str:
        """Insert n random symbols from the word at random positions.
        
        Args:
            word: Input word to transform
            n: Number of characters to insert
            
        Returns:
            Transformed word
        """
        symbols = np.array(list(word))
        new_symbols = np.random.choice(symbols, size=n)
    
        for symbol in new_symbols:
            pos = np.random.randint(0, symbols.shape[0])
            symbols = np.insert(symbols, pos, symbol)
    
        return ''.join(symbols)

    @staticmethod
    def _swap_n_pairs(word: str, n: int) -> str:
        """Swap n random pairs of symbols in the word.
        
        Args:
            word: Word to transform
            n: Number of character pairs to swap
            
        Returns:
            Transformed word
        """
        def _swap_random_pair(word: str) -> str:
            """Swap one random pair of symbols in the word."""
            symbols = np.array(list(word))
            indices = np.random.choice(symbols.shape[0], size=2, replace=False)
            symbols[indices] = symbols[indices[::-1]]
            return ''.join(symbols)
            
        for _ in range(n):
            word = _swap_random_pair(word)
        return word

    @staticmethod
    def plot_results(
        results: Dict[int, Tuple[float, float]],
        title: str,
        xlabel: str,
        ylabel: str
    ) -> None:
        """Plot the results of transformation analysis.
        
        Args:
            results: Dictionary mapping transformation parameters to (mean, variance) tuples
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
        """

        sns.set_style('darkgrid')

        x_values = list(results.keys())
        y_mean = [results[i][0] for i in x_values]
        y_var = [results[i][1] for i in x_values]
    
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
        ax1.plot(x_values, y_mean, marker='o', color='b', markersize=3, label='Mean distance')
        ax1.set_title(title)
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel('Mean')
        ax1.set_xticks(x_values)
        ax1.grid(True)
        ax1.legend()
    
        ax2.plot(x_values, y_var, marker='o', color='r', markersize=3, label='Normalized variance')
        ax2.set_title(title)
        ax2.set_xlabel(xlabel)
        ax2.set_ylabel('Normalized variance')
        ax2.set_xticks(x_values)
        ax2.grid(True)
        ax2.legend()
        plt.show()

    @staticmethod
    def _cosine_distance(vector1: np.ndarray, vector2: np.ndarray) -> float:
        """Compute cosine distance between two vectors.
        
        Args:
            vector1: First input vector
            vector2: Second input vector
            
        Returns:
            Cosine distance (1 - cosine similarity)
        """
        return 1 - np.sum(vector1 * vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))