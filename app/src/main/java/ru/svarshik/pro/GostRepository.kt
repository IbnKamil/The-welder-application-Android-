package ru.svarshik.pro

import android.content.Context
import android.net.Uri
import android.provider.OpenableColumns
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.util.UUID

internal data class UserGostDocument(
    val id: String,
    val title: String,
    val tag: String,
    val storedFileName: String,
)

internal object GostRepository {
    val defaultTags = listOf("РДС", "MIG/MAG", "TIG", "Контроль", "Трубы")

    private const val preferencesName = "user_gost_library"
    private const val documentsKey = "documents"
    private const val tagsKey = "custom_tags"

    fun loadDocuments(context: Context): List<UserGostDocument> {
        val source = preferences(context).getString(documentsKey, "[]").orEmpty()
        return runCatching {
            val array = JSONArray(source)
            buildList {
                for (index in 0 until array.length()) {
                    val item = array.getJSONObject(index)
                    val document = UserGostDocument(
                        id = item.getString("id"),
                        title = item.getString("title"),
                        tag = item.getString("tag"),
                        storedFileName = item.getString("storedFileName"),
                    )
                    if (documentFile(context, document).isFile) add(document)
                }
            }
        }.getOrDefault(emptyList())
    }

    fun loadCustomTags(context: Context): List<String> {
        val source = preferences(context).getString(tagsKey, "[]").orEmpty()
        return runCatching {
            val array = JSONArray(source)
            buildList {
                for (index in 0 until array.length()) {
                    array.optString(index).trim().takeIf(String::isNotEmpty)?.let(::add)
                }
            }
        }.getOrDefault(emptyList())
    }

    fun addTag(context: Context, rawTag: String): String {
        val tag = rawTag.trim()
        require(tag.length in 2..30) { "Название тега должно содержать от 2 до 30 символов" }
        val allTags = defaultTags + loadCustomTags(context)
        require(allTags.none { it.equals(tag, ignoreCase = true) }) {
            "Такой тег уже существует"
        }
        val tags = loadCustomTags(context) + tag
        val array = JSONArray().apply { tags.forEach(::put) }
        preferences(context).edit().putString(tagsKey, array.toString()).apply()
        return tag
    }

    fun importDocument(
        context: Context,
        uri: Uri,
        rawTitle: String,
        tag: String,
    ): UserGostDocument {
        val title = rawTitle.trim()
        require(title.length in 3..180) {
            "Название документа должно содержать от 3 до 180 символов"
        }
        require(tag.isNotBlank()) { "Выберите тег" }

        val document = UserGostDocument(
            id = UUID.randomUUID().toString(),
            title = title,
            tag = tag,
            storedFileName = "${UUID.randomUUID()}.pdf",
        )
        val destination = documentFile(context, document)
        destination.parentFile?.mkdirs()
        try {
            context.contentResolver.openInputStream(uri).use { input ->
                requireNotNull(input) { "Не удалось прочитать выбранный файл" }
                destination.outputStream().use { output -> input.copyTo(output) }
            }
            val signature = destination.inputStream().use { input ->
                ByteArray(4).also { input.read(it) }.decodeToString()
            }
            require(signature == "%PDF") { "Выбранный файл не является PDF-документом" }
        } catch (error: Throwable) {
            destination.delete()
            throw error
        }

        saveDocuments(context, loadDocuments(context) + document)
        return document
    }

    fun suggestedTitle(context: Context, uri: Uri): String {
        val name = displayName(context, uri)
        return name.substringBeforeLast('.', name).trim().ifBlank { "Новый документ" }
    }

    fun documentFile(context: Context, document: UserGostDocument): File =
        File(File(context.filesDir, "user_gosts"), document.storedFileName)

    private fun saveDocuments(context: Context, documents: List<UserGostDocument>) {
        val array = JSONArray()
        documents.forEach { document ->
            array.put(
                JSONObject()
                    .put("id", document.id)
                    .put("title", document.title)
                    .put("tag", document.tag)
                    .put("storedFileName", document.storedFileName),
            )
        }
        preferences(context).edit().putString(documentsKey, array.toString()).apply()
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
                return cursor.getString(0).orEmpty().ifBlank { "Новый документ.pdf" }
            }
        }
        return uri.lastPathSegment.orEmpty().substringAfterLast('/').ifBlank {
            "Новый документ.pdf"
        }
    }

    private fun preferences(context: Context) =
        context.getSharedPreferences(preferencesName, Context.MODE_PRIVATE)
}
