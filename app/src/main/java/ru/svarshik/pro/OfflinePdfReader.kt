package ru.svarshik.pro

import android.graphics.Bitmap
import android.graphics.Matrix
import android.graphics.pdf.PdfRenderer
import android.os.ParcelFileDescriptor
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
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.outlined.ArrowBack
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
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream

private val ReaderNavy = Color(0xFF101922)
private val ReaderOrange = Color(0xFFFF7A21)
private val ReaderBackground = Color(0xFFE8EBEE)
private val ReaderText = Color(0xFF17212B)
private val ReaderSecondary = Color(0xFF66717C)

private class PdfSession(
    val descriptor: ParcelFileDescriptor,
    val renderer: PdfRenderer,
) {
    fun close() {
        renderer.close()
        descriptor.close()
    }
}

@Composable
internal fun OfflinePdfReaderScreen(
    padding: PaddingValues,
    code: String,
    title: String,
    assetName: String? = null,
    localFilePath: String? = null,
    sourceLabel: String = "awelding.ru",
    onBack: () -> Unit,
) {
    val context = LocalContext.current
    val documentKey = localFilePath ?: assetName.orEmpty()
    var session by remember(documentKey) { mutableStateOf<PdfSession?>(null) }
    var currentPage by remember(documentKey) { mutableIntStateOf(0) }
    var pageBitmap by remember(documentKey) { mutableStateOf<Bitmap?>(null) }
    var errorMessage by remember(documentKey) { mutableStateOf<String?>(null) }
    var isLoading by remember(documentKey) { mutableStateOf(true) }
    var scale by remember(currentPage) { mutableFloatStateOf(1f) }
    var offset by remember(currentPage) { mutableStateOf(Offset.Zero) }

    BackHandler(onBack = onBack)

    LaunchedEffect(documentKey) {
        isLoading = true
        errorMessage = null
        runCatching {
            withContext(Dispatchers.IO) {
                val sourceFile = if (localFilePath != null) {
                    File(localFilePath).also {
                        require(it.isFile) { "Файл документа не найден" }
                    }
                } else {
                    val bundledAsset = requireNotNull(assetName) {
                        "Источник документа не указан"
                    }
                    val cacheDirectory = File(context.cacheDir, "offline_gosts").apply { mkdirs() }
                    File(cacheDirectory, bundledAsset).also { cachedFile ->
                        context.assets.open("gosts/$bundledAsset").use { input ->
                            FileOutputStream(cachedFile, false).use { output -> input.copyTo(output) }
                        }
                    }
                }
                val descriptor = ParcelFileDescriptor.open(
                    sourceFile,
                    ParcelFileDescriptor.MODE_READ_ONLY,
                )
                PdfSession(descriptor, PdfRenderer(descriptor))
            }
        }.onSuccess {
            session = it
        }.onFailure {
            errorMessage = "Не удалось открыть документ: ${it.localizedMessage ?: "неизвестная ошибка"}"
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
                    val matrix = Matrix().apply { postScale(renderScale, renderScale) }
                    page.render(bitmap, null, matrix, PdfRenderer.Page.RENDER_MODE_FOR_DISPLAY)
                    bitmap
                }
            }
        }.onSuccess { renderedBitmap ->
            pageBitmap?.takeIf { it !== renderedBitmap }?.recycle()
            pageBitmap = renderedBitmap
            isLoading = false
        }.onFailure {
            errorMessage = "Не удалось отобразить страницу: ${it.localizedMessage ?: "неизвестная ошибка"}"
            isLoading = false
        }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(padding)
            .background(ReaderBackground),
    ) {
        Surface(color = ReaderNavy, shadowElevation = 4.dp) {
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
                        code,
                        color = ReaderOrange,
                        fontSize = 12.sp,
                        fontWeight = FontWeight.ExtraBold,
                    )
                    Text(
                        title,
                        color = Color.White,
                        fontSize = 14.sp,
                        fontWeight = FontWeight.Bold,
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis,
                    )
                }
                Surface(
                    color = Color.White.copy(alpha = 0.1f),
                    shape = RoundedCornerShape(12.dp),
                ) {
                    Text(
                        "ОФЛАЙН",
                        modifier = Modifier.padding(horizontal = 10.dp, vertical = 6.dp),
                        color = Color.White,
                        fontSize = 9.sp,
                        fontWeight = FontWeight.Bold,
                    )
                }
                Spacer(Modifier.width(10.dp))
            }
        }

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

            if (isLoading) {
                CircularProgressIndicator(color = ReaderOrange)
            }

            errorMessage?.let { message ->
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
                        Text(
                            "Документ не открылся",
                            color = ReaderText,
                            fontWeight = FontWeight.Bold,
                        )
                        Spacer(Modifier.height(7.dp))
                        Text(
                            message,
                            color = ReaderSecondary,
                            fontSize = 12.sp,
                        )
                    }
                }
            }
        }

        Surface(color = Color.White, shadowElevation = 8.dp) {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 14.dp, vertical = 10.dp),
                horizontalAlignment = Alignment.CenterHorizontally,
            ) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.SpaceBetween,
                ) {
                    Button(
                        onClick = { currentPage-- },
                        enabled = currentPage > 0 && !isLoading,
                        colors = ButtonDefaults.buttonColors(containerColor = ReaderNavy),
                        shape = RoundedCornerShape(12.dp),
                    ) {
                        Text("Назад")
                    }
                    Column(horizontalAlignment = Alignment.CenterHorizontally) {
                        Text(
                            if (session == null) "Подготовка…" else "${currentPage + 1} / ${session!!.renderer.pageCount}",
                            color = ReaderText,
                            fontWeight = FontWeight.ExtraBold,
                            fontSize = 16.sp,
                        )
                        Text(
                            if (scale > 1f) "Масштаб ${(scale * 100).toInt()}%" else "Разведите пальцы для увеличения",
                            color = ReaderSecondary,
                            fontSize = 9.sp,
                        )
                    }
                    Button(
                        onClick = { currentPage++ },
                        enabled = session != null &&
                            currentPage < session!!.renderer.pageCount - 1 &&
                            !isLoading,
                        colors = ButtonDefaults.buttonColors(containerColor = ReaderOrange),
                        shape = RoundedCornerShape(12.dp),
                    ) {
                        Text("Далее")
                    }
                }
                Text(
                    "Источник: $sourceLabel",
                    color = ReaderSecondary,
                    fontSize = 9.sp,
                    modifier = Modifier.padding(top = 5.dp),
                )
            }
        }
    }
}
