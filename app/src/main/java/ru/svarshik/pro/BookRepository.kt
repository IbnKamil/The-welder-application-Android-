package ru.svarshik.pro

import android.content.Context
import android.net.Uri
import android.provider.OpenableColumns
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.util.UUID

internal data class UserBook(
    val id: String,
    val title: String,
    val storedFileName: String,
    val format: String,
    val bookmark: Int = 0,
)

internal object BookRepository {
    private const val preferencesName = "user_library"
    private const val booksKey = "books"
    private val supportedFormats = setOf("pdf", "epub", "txt")

    fun loadBooks(context: Context): List<UserBook> {
        val source = context
            .getSharedPreferences(preferencesName, Context.MODE_PRIVATE)
            .getString(booksKey, "[]")
            .orEmpty()

        return runCatching {
            val array = JSONArray(source)
            buildList {
                for (index in 0 until array.length()) {
                    val item = array.getJSONObject(index)
                    val book = UserBook(
                        id = item.getString("id"),
                        title = item.getString("title"),
                        storedFileName = item.getString("storedFileName"),
                        format = item.getString("format"),
                        bookmark = item.optInt("bookmark", 0),
                    )
                    if (bookFile(context, book).isFile) add(book)
                }
            }
        }.getOrDefault(emptyList())
    }

    fun importBook(context: Context, uri: Uri): UserBook {
        val originalName = displayName(context, uri)
        val format = detectFormat(context, uri, originalName)
        require(format in supportedFormats) {
            "Поддерживаются только PDF, EPUB и TXT"
        }

        val title = originalName
            .substringBeforeLast('.', originalName)
            .trim()
            .ifBlank { "Книга без названия" }
        val book = UserBook(
            id = UUID.randomUUID().toString(),
            title = title,
            storedFileName = "${UUID.randomUUID()}.$format",
            format = format,
        )

        val destination = bookFile(context, book)
        destination.parentFile?.mkdirs()
        try {
            context.contentResolver.openInputStream(uri).use { input ->
                requireNotNull(input) { "Не удалось прочитать выбранный файл" }
                destination.outputStream().use { output -> input.copyTo(output) }
            }
        } catch (error: Throwable) {
            destination.delete()
            throw error
        }

        val books = loadBooks(context) + book
        saveBooks(context, books)
        return book
    }

    fun saveBookmark(context: Context, bookId: String, bookmark: Int): UserBook? {
        var updatedBook: UserBook? = null
        val books = loadBooks(context).map { book ->
            if (book.id == bookId) {
                book.copy(bookmark = bookmark.coerceAtLeast(0)).also { updatedBook = it }
            } else {
                book
            }
        }
        saveBooks(context, books)
        return updatedBook
    }

    fun bookFile(context: Context, book: UserBook): File =
        File(File(context.filesDir, "books"), book.storedFileName)

    private fun saveBooks(context: Context, books: List<UserBook>) {
        val array = JSONArray()
        books.forEach { book ->
            array.put(
                JSONObject()
                    .put("id", book.id)
                    .put("title", book.title)
                    .put("storedFileName", book.storedFileName)
                    .put("format", book.format)
                    .put("bookmark", book.bookmark),
            )
        }
        context
            .getSharedPreferences(preferencesName, Context.MODE_PRIVATE)
            .edit()
            .putString(booksKey, array.toString())
            .apply()
    }

    private fun displayName(context: Context, uri: Uri): String {
        context.contentResolver.query(
            uri,
            arrayOf(OpenableColumns.DISPLAY_NAME),
            null,
            null,
            null,
        )?.use { cursor ->
            if (cursor.moveToFirst()) {
                return cursor.getString(0).orEmpty().ifBlank { "Книга" }
            }
        }
        return uri.lastPathSegment.orEmpty().substringAfterLast('/').ifBlank { "Книга" }
    }

    private fun detectFormat(context: Context, uri: Uri, name: String): String {
        val extension = name.substringAfterLast('.', "").lowercase()
        if (extension in supportedFormats) return extension
        return when (context.contentResolver.getType(uri)?.lowercase()) {
            "application/pdf" -> "pdf"
            "application/epub+zip" -> "epub"
            "text/plain" -> "txt"
            else -> extension
        }
    }
}
