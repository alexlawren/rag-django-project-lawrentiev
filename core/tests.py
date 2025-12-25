from django.test import TestCase
import os
import shutil
import chromadb
from chromadb.config import Settings

# Импортируем функции, которые мы хотим протестировать
from . import rag_logic


class RagLogicTests(TestCase):

    def setUp(self):
        self.test_file_path = 'test_document_for_unittest.txt'
        with open(self.test_file_path, 'w', encoding='utf-8') as f:
            f.write(
                "Разработка программного обеспечения — это сложный процесс. "
                "Он требует анализа требований и проектирования архитектуры. "
                "Тестирование является неотъемлемой частью этого процесса. "
                "Финальным этапом является развертывание."
            )

        self.test_db_path = './test_chroma_db_for_unittest'

        # При создании клиента передаем настройки, которые разрешают reset
        self.client = chromadb.PersistentClient(
            path=self.test_db_path,
            settings=Settings(allow_reset=True)
        )

        # Полностью очищаем БД перед каждым тестом для 100% изоляции
        self.client.reset()
        self.collection = self.client.get_or_create_collection(name="test_collection")

    def tearDown(self):
        if os.path.exists(self.test_file_path):
            os.remove(self.test_file_path)

        # Мы уже сбрасываем клиент в setUp, здесь reset не нужен.
        # Просто удаляем папку.
        if os.path.exists(self.test_db_path):
            shutil.rmtree(self.test_db_path, ignore_errors=True)

    def test_01_create_sentence_window_chunks(self):
        """Тестируем функцию сегментации и чанкинга."""
        chunks = rag_logic.create_sentence_window_chunks(self.test_file_path)
        self.assertIsInstance(chunks, list)
        self.assertEqual(len(chunks), 4)
        self.assertIsInstance(chunks[0], dict)
        self.assertIn('sentence', chunks[0])
        self.assertIn('window', chunks[0])

    def test_02_add_and_index_chunks(self):
        """Тестируем функцию векторизации и добавления в БД."""
        chunks = rag_logic.create_sentence_window_chunks(self.test_file_path)
        initial_count = self.collection.count()
        rag_logic.add_sentence_chunks_to_collection(self.collection, chunks)
        final_count = self.collection.count()
        self.assertEqual(initial_count, 0)
        self.assertEqual(final_count, 4)

    def test_03_search_and_rerank(self):
        """Тестируем функцию поиска и переранжирования."""
        chunks = rag_logic.create_sentence_window_chunks(self.test_file_path)
        rag_logic.add_sentence_chunks_to_collection(self.collection, chunks)
        test_query = "Что является неотъемлемой частью?"
        context = rag_logic.search_and_rerank(self.collection, test_query)
        self.assertIsInstance(context, list)
        self.assertTrue(len(context) > 0)
        # Сравниваем в нижнем регистре для надежности
        self.assertIn("тестирование", context[0].lower())

    def test_04_generate_answer(self):
        """Тестируем функцию генерации ответа."""
        test_context = ["Тестирование является неотъемлемой частью этого процесса."]
        test_query = "Что является неотъемлемой частью?"
        answer = rag_logic.generate_answer_from_context(test_context, test_query)
        self.assertIsInstance(answer, str)
        self.assertTrue(len(answer) > 0)
        # Убираем знаки препинания и приводим к нижнему регистру
        normalized_answer = answer.lower().replace('.', '').replace(',', '')
        self.assertIn("тестирован", normalized_answer)