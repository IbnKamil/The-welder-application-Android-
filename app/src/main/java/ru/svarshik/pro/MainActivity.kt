package ru.svarshik.pro

import android.Manifest
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.outlined.MenuBook
import androidx.compose.material.icons.outlined.AutoAwesome
import androidx.compose.material.icons.outlined.Business
import androidx.compose.material.icons.outlined.CameraAlt
import androidx.compose.material.icons.outlined.ChatBubbleOutline
import androidx.compose.material.icons.outlined.Description
import androidx.compose.material.icons.outlined.Groups
import androidx.compose.material.icons.outlined.Image
import androidx.compose.material.icons.outlined.NotificationsNone
import androidx.compose.material.icons.outlined.Person
import androidx.compose.material.icons.outlined.PlayCircleOutline
import androidx.compose.material.icons.outlined.Psychology
import androidx.compose.material.icons.outlined.School
import androidx.compose.material.icons.outlined.Search
import androidx.compose.material.icons.outlined.Storefront
import androidx.compose.material.icons.outlined.TaskAlt
import androidx.compose.material.icons.outlined.Tune
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.SnackbarHost
import androidx.compose.material3.SnackbarHostState
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import kotlinx.coroutines.launch

private val Navy = Color(0xFF101922)
private val DarkNavy = Color(0xFF0A1118)
private val Orange = Color(0xFFFF7A21)
private val OrangeSoft = Color(0xFFFFE8D8)
private val AppBackground = Color(0xFFF4F6F8)
private val TextPrimary = Color(0xFF17212B)
private val TextSecondary = Color(0xFF66717C)
private val Success = Color(0xFF28A879)

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            MaterialTheme(
                colorScheme = MaterialTheme.colorScheme.copy(
                    primary = Orange,
                    surface = Color.White,
                    background = AppBackground,
                    onSurface = TextPrimary,
                ),
            ) {
                SvarshikProApp()
            }
        }
    }
}

private data class AppSection(
    val title: String,
    val shortTitle: String,
    val icon: ImageVector,
)

private val sections = listOf(
    AppSection("Анализ дефектов", "Анализ", Icons.Outlined.AutoAwesome),
    AppSection("ГОСТы", "ГОСТы", Icons.Outlined.Description),
    AppSection("Библиотека", "Книги", Icons.AutoMirrored.Outlined.MenuBook),
    AppSection("Курсы", "Курсы", Icons.Outlined.School),
    AppSection("Тренажёры", "Практика", Icons.Outlined.Psychology),
    AppSection("Мессенджер", "Чаты", Icons.Outlined.ChatBubbleOutline),
    AppSection("Социальная сеть", "Сообщество", Icons.Outlined.Groups),
    AppSection("Рынок", "Рынок", Icons.Outlined.Storefront),
)

@Composable
private fun SvarshikProApp() {
    var selectedSection by remember { mutableIntStateOf(0) }
    var selectedGost by remember { mutableStateOf<DocumentItem?>(null) }
    val snackbarHostState = remember { SnackbarHostState() }
    val scope = rememberCoroutineScope()

    Scaffold(
        containerColor = AppBackground,
        snackbarHost = { SnackbarHost(snackbarHostState) },
        topBar = {
            if (selectedGost == null) {
                AppHeader(
                    section = sections[selectedSection],
                    onAction = { message -> scope.launch { snackbarHostState.showSnackbar(message) } },
                )
            }
        },
        bottomBar = {
            if (selectedGost == null) {
                AppNavigation(
                    selected = selectedSection,
                    onSelected = { selectedSection = it },
                )
            }
        },
    ) { padding ->
        selectedGost?.let { document ->
            OfflinePdfReaderScreen(
                padding = padding,
                code = document.code,
                title = document.title,
                assetName = document.assetName,
                onBack = { selectedGost = null },
            )
            return@Scaffold
        }
        when (selectedSection) {
            0 -> AnalysisScreen(padding, snackbarHostState)
            1 -> GostScreen(padding, onDocumentSelected = { selectedGost = it })
            2 -> LibraryScreen(padding)
            3 -> CoursesScreen(padding)
            4 -> TrainersScreen(padding)
            5 -> ComingSoonScreen(
                padding = padding,
                icon = Icons.Outlined.ChatBubbleOutline,
                title = "Мессенджер",
                description = "Профессиональные чаты, личные сообщения и рабочие группы появятся в следующем обновлении.",
            )
            6 -> ComingSoonScreen(
                padding = padding,
                icon = Icons.Outlined.Groups,
                title = "Сообщество",
                description = "Лента работ, обсуждения технологий и подписки на мастеров находятся в разработке.",
            )
            else -> MarketScreen(padding)
        }
    }
}

@Composable
private fun AppHeader(section: AppSection, onAction: (String) -> Unit) {
    Surface(color = Navy, shadowElevation = 4.dp) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .statusBarsPadding()
                .height(64.dp)
                .padding(horizontal = 18.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Box(
                modifier = Modifier
                    .size(38.dp)
                    .clip(RoundedCornerShape(11.dp))
                    .background(Orange),
                contentAlignment = Alignment.Center,
            ) {
                Text("С", color = Color.White, fontWeight = FontWeight.Black, fontSize = 21.sp)
            }
            Spacer(Modifier.width(12.dp))
            Column(modifier = Modifier.weight(1f)) {
                Text(
                    text = "СВАРЩИК ПРО",
                    color = Color.White,
                    fontWeight = FontWeight.ExtraBold,
                    fontSize = 17.sp,
                    letterSpacing = 0.4.sp,
                )
                Text(
                    text = section.title,
                    color = Color.White.copy(alpha = 0.62f),
                    fontSize = 12.sp,
                )
            }
            IconButton(onClick = { onAction("Поиск по разделу «${section.title}»") }) {
                Icon(Icons.Outlined.Search, contentDescription = "Поиск", tint = Color.White)
            }
            IconButton(onClick = { onAction("Новых уведомлений нет") }) {
                Icon(
                    Icons.Outlined.NotificationsNone,
                    contentDescription = "Уведомления",
                    tint = Color.White,
                )
            }
            IconButton(onClick = { onAction("Профиль пользователя") }) {
                Icon(Icons.Outlined.Person, contentDescription = "Профиль", tint = Color.White)
            }
        }
    }
}

@Composable
private fun AppNavigation(selected: Int, onSelected: (Int) -> Unit) {
    Surface(color = DarkNavy, shadowElevation = 12.dp) {
        LazyRow(
            modifier = Modifier
                .fillMaxWidth()
                .navigationBarsPadding(),
            contentPadding = PaddingValues(horizontal = 6.dp, vertical = 6.dp),
            horizontalArrangement = Arrangement.spacedBy(2.dp),
        ) {
            items(sections.size) { index ->
                val item = sections[index]
                val isSelected = index == selected
                Column(
                    modifier = Modifier
                        .width(78.dp)
                        .clip(RoundedCornerShape(12.dp))
                        .clickable { onSelected(index) }
                        .background(if (isSelected) Orange.copy(alpha = 0.14f) else Color.Transparent)
                        .padding(vertical = 7.dp, horizontal = 4.dp),
                    horizontalAlignment = Alignment.CenterHorizontally,
                ) {
                    Icon(
                        imageVector = item.icon,
                        contentDescription = item.title,
                        tint = if (isSelected) Orange else Color.White.copy(alpha = 0.55f),
                        modifier = Modifier.size(23.dp),
                    )
                    Spacer(Modifier.height(3.dp))
                    Text(
                        text = item.shortTitle,
                        color = if (isSelected) Color.White else Color.White.copy(alpha = 0.55f),
                        fontSize = 10.sp,
                        fontWeight = if (isSelected) FontWeight.Bold else FontWeight.Medium,
                        maxLines = 1,
                    )
                }
            }
        }
    }
}

@Composable
private fun AnalysisScreen(padding: PaddingValues, snackbarHostState: SnackbarHostState) {
    val scope = rememberCoroutineScope()
    val cameraPermissionLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.RequestPermission(),
    ) { granted ->
        scope.launch {
            snackbarHostState.showSnackbar(
                if (granted) "Камера готова. Модуль анализа будет подключён к AI-модели."
                else "Для съёмки сварного шва нужен доступ к камере.",
            )
        }
    }

    LazyColumn(
        modifier = Modifier
            .fillMaxSize()
            .padding(padding),
        contentPadding = PaddingValues(16.dp),
        verticalArrangement = Arrangement.spacedBy(14.dp),
    ) {
        item {
            Column {
                Text(
                    "Контроль сварного шва",
                    style = MaterialTheme.typography.headlineSmall,
                    fontWeight = FontWeight.ExtraBold,
                    color = TextPrimary,
                )
                Spacer(Modifier.height(5.dp))
                Text(
                    "Сделайте чёткий снимок — система подготовит карту дефектов и рекомендации.",
                    color = TextSecondary,
                    lineHeight = 20.sp,
                )
            }
        }
        item {
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .height(236.dp)
                    .clip(RoundedCornerShape(24.dp))
                    .background(
                        Brush.linearGradient(
                            listOf(Color(0xFF1C2A36), Color(0xFF0B1117)),
                        ),
                    ),
                contentAlignment = Alignment.Center,
            ) {
                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                    Box(
                        modifier = Modifier
                            .size(80.dp)
                            .clip(CircleShape)
                            .background(Color.White.copy(alpha = 0.08f)),
                        contentAlignment = Alignment.Center,
                    ) {
                        Icon(
                            Icons.Outlined.CameraAlt,
                            contentDescription = null,
                            tint = Orange,
                            modifier = Modifier.size(38.dp),
                        )
                    }
                    Spacer(Modifier.height(17.dp))
                    Text(
                        "Наведите камеру на шов",
                        color = Color.White,
                        fontSize = 18.sp,
                        fontWeight = FontWeight.Bold,
                    )
                    Spacer(Modifier.height(5.dp))
                    Text(
                        "Хорошее освещение · 15–30 см",
                        color = Color.White.copy(alpha = 0.6f),
                        fontSize = 13.sp,
                    )
                }
                Surface(
                    modifier = Modifier
                        .align(Alignment.TopEnd)
                        .padding(14.dp),
                    color = Success.copy(alpha = 0.18f),
                    shape = RoundedCornerShape(20.dp),
                ) {
                    Text(
                        "AI · BETA",
                        modifier = Modifier.padding(horizontal = 11.dp, vertical = 6.dp),
                        color = Color(0xFF67E0B0),
                        fontSize = 11.sp,
                        fontWeight = FontWeight.Bold,
                    )
                }
            }
        }
        item {
            Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                Button(
                    onClick = { cameraPermissionLauncher.launch(Manifest.permission.CAMERA) },
                    modifier = Modifier
                        .weight(1f)
                        .height(54.dp),
                    shape = RoundedCornerShape(15.dp),
                    colors = ButtonDefaults.buttonColors(containerColor = Orange),
                ) {
                    Icon(Icons.Outlined.CameraAlt, contentDescription = null)
                    Spacer(Modifier.width(8.dp))
                    Text("Сканировать", fontWeight = FontWeight.Bold)
                }
                Button(
                    onClick = {
                        scope.launch {
                            snackbarHostState.showSnackbar("Выбор снимка будет доступен после подключения хранилища.")
                        }
                    },
                    modifier = Modifier
                        .weight(1f)
                        .height(54.dp),
                    shape = RoundedCornerShape(15.dp),
                    colors = ButtonDefaults.buttonColors(
                        containerColor = Color.White,
                        contentColor = TextPrimary,
                    ),
                ) {
                    Icon(Icons.Outlined.Image, contentDescription = null)
                    Spacer(Modifier.width(8.dp))
                    Text("Из галереи", fontWeight = FontWeight.Bold)
                }
            }
        }
        item { SectionTitle("Что определяет система", "Все дефекты") }
        item {
            Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                DetectionCard("Трещины", "Продольные и поперечные", "01", Modifier.weight(1f))
                DetectionCard("Пористость", "Газовые включения", "02", Modifier.weight(1f))
            }
        }
        item {
            Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                DetectionCard("Подрезы", "Кромки и глубина", "03", Modifier.weight(1f))
                DetectionCard("Непровар", "Нарушение сплавления", "04", Modifier.weight(1f))
            }
        }
        item {
            InfoBanner(
                title = "Важно",
                text = "Результат AI — предварительная оценка. Ответственные соединения должны проходить аттестованный неразрушающий контроль.",
            )
        }
    }
}

@Composable
private fun DetectionCard(
    title: String,
    subtitle: String,
    number: String,
    modifier: Modifier = Modifier,
) {
    Card(
        modifier = modifier,
        shape = RoundedCornerShape(18.dp),
        colors = CardDefaults.cardColors(containerColor = Color.White),
    ) {
        Column(Modifier.padding(15.dp)) {
            Text(number, color = Orange, fontSize = 12.sp, fontWeight = FontWeight.Bold)
            Spacer(Modifier.height(17.dp))
            Text(title, fontWeight = FontWeight.Bold, color = TextPrimary)
            Spacer(Modifier.height(3.dp))
            Text(subtitle, color = TextSecondary, fontSize = 12.sp, lineHeight = 16.sp)
        }
    }
}

internal data class DocumentItem(
    val code: String,
    val title: String,
    val meta: String,
    val tag: String,
    val assetName: String,
)

@Composable
private fun GostScreen(
    padding: PaddingValues,
    onDocumentSelected: (DocumentItem) -> Unit,
) {
    var query by remember { mutableStateOf("") }
    val documents = listOf(
        DocumentItem("ГОСТ EN 1011-6—2017", "Сварка. Рекомендации по сварке металлических материалов. Часть 6. Лазерная сварка", "Офлайн · 39 страниц", "Лазер", "gost-en-1011-6-2017.pdf"),
        DocumentItem("ГОСТ 2246—70", "Проволока стальная сварочная. Технические условия", "Офлайн · 19 страниц", "Материалы", "gost-2246-70.pdf"),
        DocumentItem("ГОСТ 2601—84", "Сварка металлов. Термины и определения основных понятий", "Офлайн · 57 страниц", "Термины", "gost-2601-84.pdf"),
        DocumentItem("ГОСТ 5264—80", "Ручная дуговая сварка. Соединения сварные. Основные типы, конструктивные элементы и размеры", "Офлайн · 35 страниц", "РДС", "gost-5264-80.pdf"),
        DocumentItem("ГОСТ 8713—79", "Сварка под флюсом. Соединения сварные. Основные типы, конструктивные элементы и размеры", "Офлайн · 42 страницы", "SAW", "gost-8713-79.pdf"),
        DocumentItem("ГОСТ 10594—80", "Оборудование для дуговой, контактной, ультразвуковой сварки и плазменной обработки. Ряды параметров", "Офлайн · 3 страницы", "Оборудование", "gost-10594-80.pdf"),
        DocumentItem("ГОСТ 11533—75", "Автоматическая и полуавтоматическая дуговая сварка под флюсом. Соединения под углами", "Офлайн · 39 страниц", "SAW", "gost-11533-75.pdf"),
        DocumentItem("ГОСТ 11534—75", "Ручная дуговая сварка. Соединения сварные под острыми и тупыми углами", "Офлайн · 23 страницы", "РДС", "gost-11534-75.pdf"),
        DocumentItem("ГОСТ 11969—79", "Сварка плавлением. Основные положения и их обозначения", "Офлайн · 6 страниц", "Основы", "gost-11969-79.pdf"),
        DocumentItem("ГОСТ 14771—76", "Дуговая сварка в защитном газе. Соединения сварные", "Офлайн · 9 страниц", "MIG/MAG", "gost-14771-76.pdf"),
        DocumentItem("ГОСТ 14776—79", "Дуговая сварка. Соединения сварные точечные", "Офлайн · 12 страниц", "Дуговая", "gost-14776-79.pdf"),
        DocumentItem("ГОСТ 14806—80", "Дуговая сварка алюминия и алюминиевых сплавов в инертных газах", "Офлайн · 37 страниц", "TIG", "gost-14806-80.pdf"),
        DocumentItem("ГОСТ 15164—78", "Электрошлаковая сварка. Соединения сварные", "Офлайн · 19 страниц", "ЭШС", "gost-15164-78.pdf"),
        DocumentItem("ГОСТ 15878—79", "Контактная сварка. Соединения сварные. Конструктивные элементы и размеры", "Офлайн · 11 страниц", "Контактная", "gost-15878-79.pdf"),
        DocumentItem("ГОСТ 16037—80", "Соединения сварные стальных трубопроводов", "Офлайн · 24 страницы", "Трубы", "gost-16037-80.pdf"),
        DocumentItem("ГОСТ 16038—80", "Дуговая сварка трубопроводов из меди и медно-никелевого сплава", "Офлайн · 22 страницы", "Медь", "gost-16038-80.pdf"),
        DocumentItem("ГОСТ 19521—74", "Сварка металлов. Классификация", "Офлайн · 14 страниц", "Основы", "gost-19521-74.pdf"),
        DocumentItem("ГОСТ 20549—75", "Диффузионная сварка в вакууме рабочих элементов штампов", "Офлайн · 8 страниц", "Диффузионная", "gost-20549-75.pdf"),
        DocumentItem("ГОСТ 23055—78", "Контроль неразрушающий. Классификация сварных соединений по радиографическому контролю", "Офлайн · 8 страниц", "Контроль", "gost-23055-78.pdf"),
        DocumentItem("ГОСТ 23338—91", "Методы определения содержания диффузионного водорода в наплавленном металле и металле шва", "Офлайн · 21 страница", "Контроль", "gost-23338-91.pdf"),
        DocumentItem("ГОСТ 23518—79", "Дуговая сварка в защитных газах. Соединения под острыми и тупыми углами", "Офлайн · 28 страниц", "MIG/MAG", "gost-23518-79.pdf"),
        DocumentItem("ГОСТ 25997—83", "Сварка металлов плавлением. Статистическая оценка качества", "Офлайн · 18 страниц", "Контроль", "gost-25997-83.pdf"),
        DocumentItem("ГОСТ 27580—88", "Дуговая сварка алюминия и алюминиевых сплавов. Соединения под углами", "Офлайн · 38 страниц", "TIG", "gost-27580-88.pdf"),
        DocumentItem("ГОСТ 28915—91", "Сварка лазерная импульсная. Соединения сварные точечные", "Офлайн · 10 страниц", "Лазер", "gost-28915-91.pdf"),
        DocumentItem("ГОСТ 30430—96", "Сварка дуговая конструкционных чугунов. Требования к технологическому процессу", "Офлайн · 15 страниц", "Чугун", "gost-30430-96.pdf"),
        DocumentItem("ГОСТ 30482—97", "Сварка сталей электрошлаковая. Требования к технологическому процессу", "Офлайн · 19 страниц", "ЭШС", "gost-30482-97.pdf"),
        DocumentItem("ГОСТ 33857—2016", "Арматура трубопроводная. Сварка и контроль качества сварных соединений", "Офлайн · 88 страниц", "Трубы", "gost-33857-2016.pdf"),
        DocumentItem("ГОСТ 34061—2017", "Определение содержания водорода в наплавленном металле и металле шва", "Офлайн · 36 страниц", "Водород", "gost-34061-2017-iso3690.pdf"),
    )
    val filteredDocuments = documents.filter {
        query.isBlank() ||
            it.code.contains(query, ignoreCase = true) ||
            it.title.contains(query, ignoreCase = true) ||
            it.tag.contains(query, ignoreCase = true)
    }
    ContentList(
        padding = padding,
        title = "Нормативные документы",
        subtitle = "28 документов доступны без интернета",
        searchHint = "Найти по номеру или названию",
        searchValue = query,
        onSearchValueChange = { query = it },
    ) {
        item {
            LazyRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                items(listOf("Все", "РДС", "MIG/MAG", "TIG", "Контроль", "Трубы")) {
                    FilterChipLabel(it, selected = it == "Все")
                }
            }
        }
        items(filteredDocuments) { document ->
            DocumentCard(document, onClick = { onDocumentSelected(document) })
        }
        if (filteredDocuments.isEmpty()) {
            item {
                Text(
                    "По запросу «$query» ничего не найдено",
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(vertical = 30.dp),
                    textAlign = TextAlign.Center,
                    color = TextSecondary,
                )
            }
        }
        item {
            InfoBanner(
                "Источник и актуальность",
                "Документы загружены с awelding.ru. Перед применением проверяйте редакцию и статус в официальном фонде стандартов.",
            )
        }
    }
}

@Composable
private fun DocumentCard(document: DocumentItem, onClick: () -> Unit) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .clickable(onClick = onClick),
        shape = RoundedCornerShape(18.dp),
        colors = CardDefaults.cardColors(containerColor = Color.White),
    ) {
        Row(Modifier.padding(16.dp), verticalAlignment = Alignment.CenterVertically) {
            Box(
                modifier = Modifier
                    .size(48.dp)
                    .clip(RoundedCornerShape(13.dp))
                    .background(OrangeSoft),
                contentAlignment = Alignment.Center,
            ) {
                Icon(Icons.Outlined.Description, null, tint = Orange)
            }
            Spacer(Modifier.width(13.dp))
            Column(Modifier.weight(1f)) {
                Text(document.code, color = Orange, fontSize = 12.sp, fontWeight = FontWeight.Bold)
                Spacer(Modifier.height(3.dp))
                Text(
                    document.title,
                    color = TextPrimary,
                    fontWeight = FontWeight.Bold,
                    maxLines = 2,
                    overflow = TextOverflow.Ellipsis,
                    lineHeight = 19.sp,
                )
                Spacer(Modifier.height(5.dp))
                Text(document.meta, color = TextSecondary, fontSize = 11.sp)
            }
            Text(
                document.tag,
                modifier = Modifier
                    .clip(RoundedCornerShape(8.dp))
                    .background(AppBackground)
                    .padding(horizontal = 7.dp, vertical = 5.dp),
                color = TextSecondary,
                fontSize = 10.sp,
            )
        }
    }
}

private data class BookItem(
    val title: String,
    val author: String,
    val category: String,
    val color: Color,
)

@Composable
private fun LibraryScreen(padding: PaddingValues) {
    val books = listOf(
        BookItem("Технология электрической сварки металлов", "Б. Е. Патон", "Фундаментальная литература", Color(0xFF345B78)),
        BookItem("Сварка и свариваемые материалы", "В. М. Ямпольский", "Учебное пособие", Color(0xFF774B39)),
        BookItem("Контроль качества сварных соединений", "В. Н. Волченко", "Дефектоскопия", Color(0xFF46664B)),
        BookItem("Справочник сварщика", "Под ред. В. В. Степанова", "Практический справочник", Color(0xFF5A506F)),
    )
    ContentList(
        padding,
        "Профессиональная библиотека",
        "Учебники, справочники и работы экспертов",
        "Автор, название или тема",
    ) {
        item {
            FeaturedCard(
                eyebrow = "ВЫБОР РЕДАКЦИИ",
                title = "Основы сварочного производства",
                subtitle = "Системный курс: от металлургии до контроля качества",
                button = "Начать чтение",
            )
        }
        item { SectionTitle("Популярные книги", "Смотреть все") }
        items(books) { book -> BookCard(book) }
    }
}

@Composable
private fun BookCard(book: BookItem) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(18.dp),
        colors = CardDefaults.cardColors(containerColor = Color.White),
    ) {
        Row(Modifier.padding(14.dp), verticalAlignment = Alignment.CenterVertically) {
            Box(
                modifier = Modifier
                    .size(width = 64.dp, height = 84.dp)
                    .clip(RoundedCornerShape(8.dp))
                    .background(book.color),
                contentAlignment = Alignment.Center,
            ) {
                Icon(Icons.AutoMirrored.Outlined.MenuBook, null, tint = Color.White.copy(alpha = 0.85f))
            }
            Spacer(Modifier.width(14.dp))
            Column(Modifier.weight(1f)) {
                Text(book.category.uppercase(), color = Orange, fontSize = 10.sp, fontWeight = FontWeight.Bold)
                Spacer(Modifier.height(5.dp))
                Text(book.title, fontWeight = FontWeight.Bold, color = TextPrimary, lineHeight = 19.sp)
                Spacer(Modifier.height(6.dp))
                Text(book.author, color = TextSecondary, fontSize = 12.sp)
            }
            Icon(Icons.Outlined.PlayCircleOutline, "Открыть", tint = TextSecondary)
        }
    }
}

@Composable
private fun CoursesScreen(padding: PaddingValues) {
    ContentList(
        padding,
        "Обучение",
        "Курсы для начинающих и практикующих сварщиков",
        "Найти курс или преподавателя",
    ) {
        item {
            FeaturedCard(
                eyebrow = "БЕСПЛАТНЫЙ СТАРТ",
                title = "Безопасность сварочных работ",
                subtitle = "12 уроков · тестирование · сертификат",
                button = "Пройти курс",
            )
        }
        item { SectionTitle("Рекомендуем", "Все курсы") }
        item {
            CourseCard("Ручная дуговая сварка: уровень 1", "НИТУ МИСИС", "24 урока", 0.72f)
        }
        item {
            CourseCard("MIG/MAG: настройка и техника", "Учебный центр «Металл»", "18 уроков", 0.38f)
        }
        item {
            CourseCard("Чтение сварочных чертежей", "МГТУ им. Н. Э. Баумана", "15 уроков", 0f)
        }
    }
}

@Composable
private fun CourseCard(title: String, provider: String, lessons: String, progress: Float) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(18.dp),
        colors = CardDefaults.cardColors(containerColor = Color.White),
    ) {
        Column(Modifier.padding(17.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Box(
                    Modifier
                        .size(44.dp)
                        .clip(RoundedCornerShape(12.dp))
                        .background(Navy),
                    contentAlignment = Alignment.Center,
                ) {
                    Icon(Icons.Outlined.School, null, tint = Orange)
                }
                Spacer(Modifier.width(12.dp))
                Column(Modifier.weight(1f)) {
                    Text(title, fontWeight = FontWeight.Bold, color = TextPrimary, lineHeight = 19.sp)
                    Spacer(Modifier.height(4.dp))
                    Text(provider, color = TextSecondary, fontSize = 12.sp)
                }
            }
            Spacer(Modifier.height(15.dp))
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text(lessons, color = TextSecondary, fontSize = 12.sp, modifier = Modifier.weight(1f))
                Text(
                    if (progress > 0f) "${(progress * 100).toInt()}%" else "Не начат",
                    color = if (progress > 0f) Orange else TextSecondary,
                    fontWeight = FontWeight.Bold,
                    fontSize = 12.sp,
                )
            }
            if (progress > 0f) {
                Spacer(Modifier.height(8.dp))
                Box(
                    Modifier
                        .fillMaxWidth()
                        .height(5.dp)
                        .clip(CircleShape)
                        .background(AppBackground),
                ) {
                    Box(
                        Modifier
                            .fillMaxWidth(progress)
                            .height(5.dp)
                            .background(Orange),
                    )
                }
            }
        }
    }
}

@Composable
private fun TrainersScreen(padding: PaddingValues) {
    ContentList(
        padding,
        "Тренажёры",
        "Закрепляйте знания на практических заданиях",
        "Поиск упражнения",
    ) {
        item {
            Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                StatCard("12", "выполнено", Icons.Outlined.TaskAlt, Modifier.weight(1f))
                StatCard("7", "дней подряд", Icons.Outlined.AutoAwesome, Modifier.weight(1f))
                StatCard("84%", "точность", Icons.Outlined.Tune, Modifier.weight(1f))
            }
        }
        item { SectionTitle("Продолжить тренировку", "Мой прогресс") }
        item {
            TrainerCard(
                "Выбор режима сварки",
                "Подберите ток, напряжение и скорость подачи",
                "Средний",
                8,
            )
        }
        item {
            TrainerCard(
                "Поиск дефектов",
                "Определите нарушение по фотографии шва",
                "Продвинутый",
                12,
            )
        }
        item {
            TrainerCard(
                "Условные обозначения",
                "Прочитайте обозначение соединения на чертеже",
                "Начальный",
                10,
            )
        }
    }
}

@Composable
private fun StatCard(value: String, label: String, icon: ImageVector, modifier: Modifier) {
    Card(
        modifier = modifier,
        shape = RoundedCornerShape(16.dp),
        colors = CardDefaults.cardColors(containerColor = Color.White),
    ) {
        Column(Modifier.padding(12.dp)) {
            Icon(icon, null, tint = Orange, modifier = Modifier.size(20.dp))
            Spacer(Modifier.height(10.dp))
            Text(value, fontWeight = FontWeight.ExtraBold, fontSize = 20.sp, color = TextPrimary)
            Text(label, color = TextSecondary, fontSize = 10.sp, maxLines = 1)
        }
    }
}

@Composable
private fun TrainerCard(title: String, text: String, level: String, tasks: Int) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(18.dp),
        colors = CardDefaults.cardColors(containerColor = Color.White),
    ) {
        Row(Modifier.padding(17.dp), verticalAlignment = Alignment.CenterVertically) {
            Box(
                Modifier
                    .size(50.dp)
                    .clip(CircleShape)
                    .background(OrangeSoft),
                contentAlignment = Alignment.Center,
            ) {
                Icon(Icons.Outlined.Psychology, null, tint = Orange)
            }
            Spacer(Modifier.width(13.dp))
            Column(Modifier.weight(1f)) {
                Text(title, fontWeight = FontWeight.Bold, color = TextPrimary)
                Spacer(Modifier.height(4.dp))
                Text(text, color = TextSecondary, fontSize = 12.sp, lineHeight = 16.sp)
                Spacer(Modifier.height(8.dp))
                Text("$level · $tasks заданий", color = Orange, fontSize = 11.sp, fontWeight = FontWeight.Bold)
            }
            Icon(Icons.Outlined.PlayCircleOutline, "Начать", tint = Orange, modifier = Modifier.size(30.dp))
        }
    }
}

@Composable
private fun MarketScreen(padding: PaddingValues) {
    ContentList(
        padding,
        "Рынок",
        "Работа, специалисты, оборудование и партнёрства",
        "Вакансия, специалист или оборудование",
    ) {
        item {
            LazyRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                items(listOf("Все", "Вакансии", "Портфолио", "Оборудование", "Предприятия")) {
                    FilterChipLabel(it, selected = it == "Все")
                }
            }
        }
        item {
            MarketCard(
                type = "ВАКАНСИЯ",
                title = "Сварщик НАКС НГДО",
                company = "ПромСтройМонтаж · Екатеринбург",
                detail = "от 180 000 ₽ · Вахта",
                icon = Icons.Outlined.Business,
            )
        }
        item {
            MarketCard(
                type = "ПАРТНЁРСТВО",
                title = "Подрядчик на металлоконструкции",
                company = "УралМашЗавод · Челябинск",
                detail = "Приём заявок до 28 августа",
                icon = Icons.Outlined.Groups,
            )
        }
        item {
            MarketCard(
                type = "ОБОРУДОВАНИЕ",
                title = "Сварочный полуавтомат 500 А",
                company = "ООО «Техносвар» · Москва",
                detail = "245 000 ₽ · Новый",
                icon = Icons.Outlined.Storefront,
            )
        }
    }
}

@Composable
private fun MarketCard(
    type: String,
    title: String,
    company: String,
    detail: String,
    icon: ImageVector,
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(18.dp),
        colors = CardDefaults.cardColors(containerColor = Color.White),
    ) {
        Column(Modifier.padding(17.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Box(
                    Modifier
                        .size(46.dp)
                        .clip(RoundedCornerShape(12.dp))
                        .background(Navy),
                    contentAlignment = Alignment.Center,
                ) {
                    Icon(icon, null, tint = Orange)
                }
                Spacer(Modifier.width(12.dp))
                Column(Modifier.weight(1f)) {
                    Text(type, color = Orange, fontSize = 10.sp, fontWeight = FontWeight.ExtraBold)
                    Spacer(Modifier.height(4.dp))
                    Text(title, color = TextPrimary, fontWeight = FontWeight.Bold, fontSize = 17.sp)
                }
            }
            Spacer(Modifier.height(14.dp))
            HorizontalDivider(color = AppBackground)
            Spacer(Modifier.height(12.dp))
            Text(company, color = TextSecondary, fontSize = 13.sp)
            Spacer(Modifier.height(5.dp))
            Text(detail, color = TextPrimary, fontWeight = FontWeight.Bold, fontSize = 13.sp)
        }
    }
}

@Composable
private fun ComingSoonScreen(
    padding: PaddingValues,
    icon: ImageVector,
    title: String,
    description: String,
) {
    Box(
        modifier = Modifier
            .fillMaxSize()
            .padding(padding)
            .padding(28.dp),
        contentAlignment = Alignment.Center,
    ) {
        Column(horizontalAlignment = Alignment.CenterHorizontally) {
            Box(
                Modifier
                    .size(96.dp)
                    .clip(CircleShape)
                    .background(OrangeSoft),
                contentAlignment = Alignment.Center,
            ) {
                Icon(icon, null, tint = Orange, modifier = Modifier.size(44.dp))
            }
            Spacer(Modifier.height(24.dp))
            Text(title, fontWeight = FontWeight.ExtraBold, fontSize = 25.sp, color = TextPrimary)
            Spacer(Modifier.height(10.dp))
            Text(
                description,
                color = TextSecondary,
                textAlign = TextAlign.Center,
                lineHeight = 21.sp,
            )
            Spacer(Modifier.height(20.dp))
            Surface(color = Navy, shape = RoundedCornerShape(20.dp)) {
                Text(
                    "СКОРО",
                    modifier = Modifier.padding(horizontal = 17.dp, vertical = 8.dp),
                    color = Color.White,
                    fontSize = 11.sp,
                    fontWeight = FontWeight.Bold,
                    letterSpacing = 1.sp,
                )
            }
        }
    }
}

@Composable
private fun ContentList(
    padding: PaddingValues,
    title: String,
    subtitle: String,
    searchHint: String,
    searchValue: String? = null,
    onSearchValueChange: ((String) -> Unit)? = null,
    content: androidx.compose.foundation.lazy.LazyListScope.() -> Unit,
) {
    LazyColumn(
        modifier = Modifier
            .fillMaxSize()
            .padding(padding),
        contentPadding = PaddingValues(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        item {
            Text(title, fontWeight = FontWeight.ExtraBold, fontSize = 24.sp, color = TextPrimary)
            Spacer(Modifier.height(4.dp))
            Text(subtitle, color = TextSecondary, fontSize = 14.sp)
        }
        item {
            if (searchValue != null && onSearchValueChange != null) {
                OutlinedTextField(
                    value = searchValue,
                    onValueChange = onSearchValueChange,
                    modifier = Modifier.fillMaxWidth(),
                    placeholder = { Text(searchHint, fontSize = 13.sp) },
                    leadingIcon = {
                        Icon(
                            Icons.Outlined.Search,
                            contentDescription = null,
                            tint = TextSecondary,
                        )
                    },
                    singleLine = true,
                    shape = RoundedCornerShape(15.dp),
                )
            } else {
                Surface(
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(50.dp),
                    color = Color.White,
                    shape = RoundedCornerShape(15.dp),
                ) {
                    Row(
                        modifier = Modifier.padding(horizontal = 15.dp),
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        Icon(Icons.Outlined.Search, null, tint = TextSecondary, modifier = Modifier.size(21.dp))
                        Spacer(Modifier.width(10.dp))
                        Text(searchHint, color = TextSecondary, fontSize = 13.sp)
                    }
                }
            }
        }
        content()
    }
}

@Composable
private fun FilterChipLabel(text: String, selected: Boolean) {
    Surface(
        shape = RoundedCornerShape(20.dp),
        color = if (selected) Navy else Color.White,
        modifier = Modifier.clickable { },
    ) {
        Text(
            text,
            modifier = Modifier.padding(horizontal = 14.dp, vertical = 9.dp),
            color = if (selected) Color.White else TextSecondary,
            fontSize = 12.sp,
            fontWeight = if (selected) FontWeight.Bold else FontWeight.Medium,
        )
    }
}

@Composable
private fun FeaturedCard(
    eyebrow: String,
    title: String,
    subtitle: String,
    button: String,
) {
    Card(
        shape = RoundedCornerShape(22.dp),
        colors = CardDefaults.cardColors(containerColor = Navy),
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .background(Brush.linearGradient(listOf(Navy, Color(0xFF263A49))))
                .padding(21.dp),
        ) {
            Text(eyebrow, color = Orange, fontSize = 10.sp, fontWeight = FontWeight.ExtraBold, letterSpacing = 1.sp)
            Spacer(Modifier.height(12.dp))
            Text(title, color = Color.White, fontSize = 21.sp, fontWeight = FontWeight.ExtraBold, lineHeight = 25.sp)
            Spacer(Modifier.height(7.dp))
            Text(subtitle, color = Color.White.copy(alpha = 0.65f), fontSize = 13.sp, lineHeight = 18.sp)
            Spacer(Modifier.height(18.dp))
            Button(
                onClick = { },
                colors = ButtonDefaults.buttonColors(containerColor = Orange),
                shape = RoundedCornerShape(12.dp),
                contentPadding = PaddingValues(horizontal = 18.dp, vertical = 10.dp),
            ) {
                Text(button, fontWeight = FontWeight.Bold)
            }
        }
    }
}

@Composable
private fun SectionTitle(title: String, action: String) {
    Row(verticalAlignment = Alignment.CenterVertically) {
        Text(
            title,
            modifier = Modifier.weight(1f),
            fontWeight = FontWeight.ExtraBold,
            fontSize = 18.sp,
            color = TextPrimary,
        )
        TextButton(onClick = { }) {
            Text(action, color = Orange, fontSize = 12.sp, fontWeight = FontWeight.Bold)
        }
    }
}

@Composable
private fun InfoBanner(title: String, text: String) {
    Surface(
        color = OrangeSoft,
        shape = RoundedCornerShape(16.dp),
    ) {
        Row(Modifier.padding(15.dp)) {
            Icon(Icons.Outlined.AutoAwesome, null, tint = Orange, modifier = Modifier.size(21.dp))
            Spacer(Modifier.width(11.dp))
            Column {
                Text(title, color = TextPrimary, fontWeight = FontWeight.Bold, fontSize = 13.sp)
                Spacer(Modifier.height(3.dp))
                Text(text, color = TextSecondary, fontSize = 12.sp, lineHeight = 17.sp)
            }
        }
    }
}
