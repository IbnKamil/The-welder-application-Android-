package ru.svarshik.pro

import android.graphics.Bitmap
import android.graphics.Matrix
import android.graphics.pdf.PdfRenderer
import android.os.ParcelFileDescriptor
import android.text.Html
import android.util.Xml
import androidx.activity.compose.BackHandler
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.detectTransformGestures
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.outlined.ArrowBack
import androidx.compose.material.icons.outlined.BookmarkAdd
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.withContext
import org.xmlpull.v1.XmlPullParser
import java.io.File
import java.net.URLDecoder
import java.nio.charset.Charset
import java.util.zip.ZipFile

private val BookNavy = Color(0xFF101922)
private val BookOrange = Color(0xFFFF7A21)
private val BookBackground = Color(0xFFF2F0E9)
private val BookText = Color(0xFF24221E)
private val BookSecondary = Color(0xFF716C63)

@Composable
internal fun UserBookReaderScreen(
    padding: PaddingValues,
    book: UserBook,
    onBack: () -> Unit,
    onBookmark: (Int) -> Unit,
) {
    BackHandler(onBack = onBack)
    if (book.format == "pdf") {
        UserPdfReader(padding, book, onBack, onBookmark)
    } else {
        UserTextReader(padding, book, onBack, onBookmark)
    }
}

@Composable
private fun ReaderHeader(
    title: String,
    format: String,
    bookmarkSaved: Boolean,
    onBack: () -> Unit,
    onBookmark: () -> Unit,
) {
    Surface(color = BookNavy, shadowElevation = 4.dp) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .height(68.dp)
                .padding(horizontal = 6.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            IconButton(onClick = onBack) {
                Icon(
                    Icons.AutoMirrored.Outlined.ArrowBack,
                    contentDescription = "Назад",
                    tint = Color.White,
                )
            }
            Column(modifier = Modifier.weight(1f)) {
                Text(
                    title,
                    color = Color.White,
                    fontWeight = FontWeight.Bold,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis,
                )
                Text(
                    if (bookmarkSaved) "Закладка сохранена" else format.uppercase(),
                    color = if (bookmarkSaved) Color(0xFF67E0B0) else Color.White.copy(alpha = 0.58f),
                    fontSize = 10.sp,
                    fontWeight = FontWeight.Bold,
                )
            }
            IconButton(onClick = onBookmark) {
                Icon(
                    Icons.Outlined.BookmarkAdd,
                    contentDescription = "Поставить закладку",
                    tint = BookOrange,
                )
            }
        }
    }
}

private class UserPdfSession(
    val descriptor: ParcelFileDescriptor,
    val renderer: PdfRenderer,
) {
    fun close() {
        renderer.close()
        descriptor.close()
    }
}

@Composable
private fun UserPdfReader(
    padding: PaddingValues,
    book: UserBook,
    onBack: () -> Unit,
    onBookmark: (Int) -> Unit,
) {
    val context = LocalContext.current
    var session by remember(book.id) { mutableStateOf<UserPdfSession?>(null) }
    var currentPage by remember(book.id) { mutableIntStateOf(book.bookmark) }
    var pageBitmap by remember(book.id) { mutableStateOf<Bitmap?>(null) }
    var errorMessage by remember(book.id) { mutableStateOf<String?>(null) }
    var isLoading by remember(book.id) { mutableStateOf(true) }
    var bookmarkSaved by remember(book.id) { mutableStateOf(false) }
    var scale by remember(currentPage) { mutableFloatStateOf(1f) }
    var offset by remember(currentPage) { mutableStateOf(Offset.Zero) }

    LaunchedEffect(book.id) {
        runCatching {
            withContext(Dispatchers.IO) {
                val descriptor = ParcelFileDescriptor.open(
                    BookRepository.bookFile(context, book),
                    ParcelFileDescriptor.MODE_READ_ONLY,
                )
                UserPdfSession(descriptor, PdfRenderer(descriptor))
            }
        }.onSuccess {
            currentPage = currentPage.coerceIn(0, (it.renderer.pageCount - 1).coerceAtLeast(0))
            session = it
        }.onFailure {
            errorMessage = "Не удалось открыть PDF: ${it.localizedMessage ?: "неизвестная ошибка"}"
            isLoading = false
        }
    }

    DisposableEffect(session) {
        val activeSession = session
        onDispose {
            pageBitmap?.recycle()
            activeSession?.close()
        }
    }

    LaunchedEffect(session, currentPage) {
        val activeSession = session ?: return@LaunchedEffect
        isLoading = true
        runCatching {
            withContext(Dispatchers.IO) {
                activeSession.renderer.openPage(currentPage).use { page ->
                    val renderScale = 2f
                    val bitmap = Bitmap.createBitmap(
                        (page.width * renderScale).toInt(),
                        (page.height * renderScale).toInt(),
                        Bitmap.Config.ARGB_8888,
                    )
                    bitmap.eraseColor(android.graphics.Color.WHITE)
                    page.render(
                        bitmap,
                        null,
                        Matrix().apply { postScale(renderScale, renderScale) },
                        PdfRenderer.Page.RENDER_MODE_FOR_DISPLAY,
                    )
                    bitmap
                }
            }
        }.onSuccess { bitmap ->
            pageBitmap?.takeIf { it !== bitmap }?.recycle()
            pageBitmap = bitmap
            isLoading = false
        }.onFailure {
            errorMessage = "Не удалось показать страницу: ${it.localizedMessage ?: "неизвестная ошибка"}"
            isLoading = false
        }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(padding)
            .background(Color(0xFFE8EBEE)),
    ) {
        ReaderHeader(
            title = book.title,
            format = book.format,
            bookmarkSaved = bookmarkSaved,
            onBack = onBack,
            onBookmark = {
                onBookmark(currentPage)
                bookmarkSaved = true
            },
        )
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .weight(1f)
                .padding(10.dp)
                .background(Color(0xFFD9DDE1), RoundedCornerShape(12.dp))
                .pointerInput(currentPage) {
                    detectTransformGestures { _, pan, zoom, _ ->
                        val updatedScale = (scale * zoom).coerceIn(1f, 4f)
                        scale = updatedScale
                        offset = if (updatedScale == 1f) Offset.Zero else offset + pan
                    }
                },
            contentAlignment = Alignment.Center,
        ) {
            pageBitmap?.let { bitmap ->
                Image(
                    bitmap = bitmap.asImageBitmap(),
                    contentDescription = "Страница ${currentPage + 1}",
                    modifier = Modifier
                        .fillMaxSize()
                        .graphicsLayer(
                            scaleX = scale,
                            scaleY = scale,
                            translationX = offset.x,
                            translationY = offset.y,
                        ),
                    contentScale = ContentScale.Fit,
                )
            }
            if (isLoading) CircularProgressIndicator(color = BookOrange)
            errorMessage?.let { ReaderError(it) }
        }
        Surface(color = Color.White, shadowElevation = 8.dp) {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 14.dp, vertical = 10.dp),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.SpaceBetween,
            ) {
                Button(
                    onClick = {
                        currentPage--
                        bookmarkSaved = false
                    },
                    enabled = currentPage > 0 && !isLoading,
                    colors = ButtonDefaults.buttonColors(containerColor = BookNavy),
                    shape = RoundedCornerShape(12.dp),
                ) {
                    Text("Назад")
                }
                Text(
                    if (session == null) "Подготовка…" else "${currentPage + 1} / ${session!!.renderer.pageCount}",
                    color = BookText,
                    fontWeight = FontWeight.ExtraBold,
                )
                Button(
                    onClick = {
                        currentPage++
                        bookmarkSaved = false
                    },
                    enabled = session != null &&
                        currentPage < session!!.renderer.pageCount - 1 &&
                        !isLoading,
                    colors = ButtonDefaults.buttonColors(containerColor = BookOrange),
                    shape = RoundedCornerShape(12.dp),
                ) {
                    Text("Далее")
                }
            }
        }
    }
}

@Composable
private fun UserTextReader(
    padding: PaddingValues,
    book: UserBook,
    onBack: () -> Unit,
    onBookmark: (Int) -> Unit,
) {
    val context = LocalContext.current
    val scrollState = rememberScrollState()
    var content by remember(book.id) { mutableStateOf<String?>(null) }
    var errorMessage by remember(book.id) { mutableStateOf<String?>(null) }
    var fontSize by remember(book.id) { mutableFloatStateOf(18f) }
    var bookmarkSaved by remember(book.id) { mutableStateOf(false) }

    LaunchedEffect(book.id) {
        runCatching {
            withContext(Dispatchers.IO) {
                val file = BookRepository.bookFile(context, book)
                if (book.format == "epub") extractEpubText(file) else readTextFile(file)
            }
        }.onSuccess {
            content = it
            delay(100)
            scrollState.scrollTo(book.bookmark.coerceAtMost(scrollState.maxValue))
        }.onFailure {
            errorMessage = "Не удалось прочитать книгу: ${it.localizedMessage ?: "неизвестная ошибка"}"
        }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(padding)
            .background(BookBackground),
    ) {
        ReaderHeader(
            title = book.title,
            format = book.format,
            bookmarkSaved = bookmarkSaved,
            onBack = onBack,
            onBookmark = {
                onBookmark(scrollState.value)
                bookmarkSaved = true
            },
        )
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .background(Color.White)
                .padding(horizontal = 16.dp, vertical = 7.dp),
            horizontalArrangement = Arrangement.End,
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Text("Размер текста", color = BookSecondary, fontSize = 11.sp)
            Spacer(Modifier.width(12.dp))
            Button(
                onClick = { fontSize = (fontSize - 2f).coerceAtLeast(14f) },
                contentPadding = PaddingValues(horizontal = 12.dp, vertical = 4.dp),
                colors = ButtonDefaults.buttonColors(containerColor = BookNavy),
            ) {
                Text("A−")
            }
            Spacer(Modifier.width(7.dp))
            Button(
                onClick = { fontSize = (fontSize + 2f).coerceAtMost(32f) },
                contentPadding = PaddingValues(horizontal = 12.dp, vertical = 4.dp),
                colors = ButtonDefaults.buttonColors(containerColor = BookOrange),
            ) {
                Text("A+")
            }
        }
        Box(
            modifier = Modifier
                .fillMaxSize()
                .padding(horizontal = 8.dp),
            contentAlignment = Alignment.Center,
        ) {
            content?.let { text ->
                Text(
                    text = text,
                    modifier = Modifier
                        .fillMaxSize()
                        .verticalScroll(scrollState)
                        .padding(horizontal = 17.dp, vertical = 22.dp),
                    color = BookText,
                    fontSize = fontSize.sp,
                    lineHeight = (fontSize * 1.55f).sp,
                    fontFamily = FontFamily.Serif,
                )
            } ?: errorMessage?.let { ReaderError(it) }
                ?: CircularProgressIndicator(color = BookOrange)
        }
    }
}

@Composable
private fun ReaderError(message: String) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .padding(22.dp),
        colors = CardDefaults.cardColors(containerColor = Color.White),
        shape = RoundedCornerShape(16.dp),
    ) {
        Column(
            modifier = Modifier.padding(20.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {
            Text("Книга не открылась", color = BookText, fontWeight = FontWeight.Bold)
            Spacer(Modifier.height(7.dp))
            Text(message, color = BookSecondary, fontSize = 12.sp)
        }
    }
}

private fun readTextFile(file: File): String {
    val bytes = file.readBytes()
    val utf8 = bytes.toString(Charsets.UTF_8)
    return if ('\uFFFD' in utf8) {
        bytes.toString(Charset.forName("windows-1251"))
    } else {
        utf8
    }
}

private fun extractEpubText(file: File): String {
    ZipFile(file).use { archive ->
        val rootFile = archive.getEntry("META-INF/container.xml")
            ?.let { entry ->
                archive.getInputStream(entry).use { input ->
                    val parser = Xml.newPullParser().apply {
                        setInput(input, "UTF-8")
                    }
                    var path: String? = null
                    while (parser.eventType != XmlPullParser.END_DOCUMENT && path == null) {
                        if (parser.eventType == XmlPullParser.START_TAG && parser.name == "rootfile") {
                            path = parser.getAttributeValue(null, "full-path")
                        }
                        parser.next()
                    }
                    path
                }
            }

        val orderedEntries = if (rootFile != null) {
            epubSpineEntries(archive, rootFile)
        } else {
            emptyList()
        }
        val entries = orderedEntries.ifEmpty {
            archive.entries().asSequence()
                .filter {
                    !it.isDirectory &&
                        (it.name.endsWith(".xhtml", true) || it.name.endsWith(".html", true))
                }
                .map { it.name }
                .sorted()
                .toList()
        }

        val text = entries.joinToString("\n\n") { entryName ->
            archive.getEntry(entryName)?.let { entry ->
                archive.getInputStream(entry).bufferedReader().use { reader ->
                    Html.fromHtml(reader.readText(), Html.FROM_HTML_MODE_LEGACY)
                        .toString()
                        .trim()
                }
            }.orEmpty()
        }.trim()
        require(text.isNotBlank()) { "В EPUB не найден читаемый текст" }
        return text
    }
}

private fun epubSpineEntries(archive: ZipFile, rootFile: String): List<String> {
    val opfEntry = archive.getEntry(rootFile) ?: return emptyList()
    val manifest = mutableMapOf<String, String>()
    val spine = mutableListOf<String>()
    archive.getInputStream(opfEntry).use { input ->
        val parser = Xml.newPullParser().apply { setInput(input, "UTF-8") }
        while (parser.eventType != XmlPullParser.END_DOCUMENT) {
            if (parser.eventType == XmlPullParser.START_TAG) {
                when (parser.name) {
                    "item" -> {
                        val id = parser.getAttributeValue(null, "id")
                        val href = parser.getAttributeValue(null, "href")
                        if (id != null && href != null) manifest[id] = href
                    }
                    "itemref" -> parser.getAttributeValue(null, "idref")?.let(spine::add)
                }
            }
            parser.next()
        }
    }
    val basePath = rootFile.substringBeforeLast('/', "")
    return spine.mapNotNull(manifest::get).map { href ->
        val decoded = URLDecoder.decode(href.substringBefore('#'), "UTF-8")
        if (basePath.isBlank()) decoded else "$basePath/$decoded"
    }
}
