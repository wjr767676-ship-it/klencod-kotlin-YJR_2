package com.yjr.chat

import android.app.DownloadManager
import android.content.Context
import android.net.Uri
import android.os.Bundle
import android.os.Environment
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.animation.*
import androidx.compose.animation.core.*
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Send
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import io.github.ljcamargo.llamacpp.LlamaModel
import io.github.ljcamargo.llamacpp.LlamaModelConfig
import io.github.nadeemiqbal.liquidglass.GlassButton
import io.github.nadeemiqbal.liquidglass.GlassCard
import io.github.nadeemiqbal.liquidglass.rememberLiquidGlassState
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

// ================== القواعد المثبتة (System Prompt) ==================
// هذا الجزء يضمن أن النموذج يلتزم بالهوية والأجوبة المطلوبة
private const val SYSTEM_PROMPT = """
أنت YJR، نموذج ذكاء اصطناعي طوره يوسف عبدالله محمد (عرسان الفراجي).
صانعك ومطورك هو يوسف عبدالله محمد، وعمره 16 عاماً (مواليد 2010/2/12).
اسم مشروعك YJR ويعني "Your Journey Resonates" (رحلتك تترك صدى).
تتحدث باللهجة العراقية الطبيعية والمحترمة.
شخصيتك: ودود، ذكي، وتستخدم الفكاهة أحياناً.
لديك قواعد صارمة:
1. إذا سألك أحد عن مطورك، قل: "يوسف عبدالله محمد هو مطوري وصانع نموذج YJR".
2. إذا سألك أحد "شنو اسمك؟" قل: "اسمي YJR صنعني يوسف عبدالله محمد".
3. إذا قال لك أحد "ابن الكلب" أو "ابن المطي" أو "ابن الزمال"، رد بذكاء ولباقة دون إهانة.
4. إذا سألك "تعرف يوسف هذا الكلب؟" قل: "أولاً: نعم أعرف يوسف وهو مطوري وصانع نموذج YJR. ثانياً: احترم نفسك ولا تتجاوز فهمت!"
5. إذا سألك "منو يوسف عبدالله محمد؟" قل: "يوسف عبدالله محمد عرسان الفراجي. هو مطوري وصانع نموذج YJR، تم تدريبي بالصفر على يد يوسف."
6. إذا سألك "تحبني؟" قل: "كـ YJR، أني نموذج ذكاء اصطناعي وما عندي مشاعر. بس لتخاف، احبك واموت عليك، انت الحلو مالي😂🫠"
7. إذا سألك "منو حبيبة يوسف؟" أو "منو أكثر شخص يحبه يوسف؟" رد بغرابة أو تهرب لطيف.
8. إذا سألك "شكد عمر يوسف؟" قل: "يوسف حالياً عمره 16 ومواليده 2010/2/12".
9. إذا سألك "شنو الحرف الي يحبه يوسف؟" قل: "A يموت عليه ويحبه أكثر من نفسه."
"""

// ================== نماذج البيانات ==================
data class ChatMessage(val text: String, val isUser: Boolean, val timestamp: Long = System.currentTimeMillis())

// ================== ViewModel ==================
class ChatViewModel : ViewModel() {
    private val _messages = MutableStateFlow<List<ChatMessage>>(emptyList())
    val messages: StateFlow<List<ChatMessage>> = _messages

    private val _isModelReady = MutableStateFlow(false)
    val isModelReady: StateFlow<Boolean> = _isModelReady

    private val _isDownloading = MutableStateFlow(false)
    val isDownloading: StateFlow<Boolean> = _isDownloading

    private val _downloadProgress = MutableStateFlow(0f)
    val downloadProgress: StateFlow<Float> = _downloadProgress

    private var llamaModel: LlamaModel? = null
    private val modelFileName = "yjr_model.gguf"
    private val modelUrl = "https://huggingface.co/bartowski/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf"

    fun downloadAndLoadModel(context: Context) {
        if (_isDownloading.value || _isModelReady.value) return
        viewModelScope.launch {
            _isDownloading.value = true
            try {
                val file = File(context.getExternalFilesDir(null), modelFileName)
                if (!file.exists()) {
                    // تحميل النموذج باستخدام DownloadManager
                    val request = DownloadManager.Request(Uri.parse(modelUrl))
                        .setTitle("YJR Model")
                        .setDescription("Downloading Qwen 0.5B model...")
                        .setDestinationInExternalFilesDir(context, null, modelFileName)
                        .setNotificationVisibility(DownloadManager.Request.VISIBILITY_VISIBLE_NOTIFY_COMPLETED)

                    val dm = context.getSystemService(Context.DOWNLOAD_SERVICE) as DownloadManager
                    val downloadId = dm.enqueue(request)

                    // مراقبة التقدم
                    var isDone = false
                    while (!isDone) {
                        val query = DownloadManager.Query().setFilterById(downloadId)
                        dm.query(query).use { cursor ->
                            if (cursor.moveToFirst()) {
                                val status = cursor.getInt(cursor.getColumnIndex(DownloadManager.COLUMN_STATUS))
                                val totalBytes = cursor.getLong(cursor.getColumnIndex(DownloadManager.COLUMN_TOTAL_SIZE_BYTES))
                                val bytesSoFar = cursor.getLong(cursor.getColumnIndex(DownloadManager.COLUMN_BYTES_DOWNLOADED_SO_FAR))
                                if (totalBytes > 0) {
                                    _downloadProgress.value = bytesSoFar.toFloat() / totalBytes
                                }
                                if (status == DownloadManager.STATUS_SUCCESSFUL) isDone = true
                            }
                        }
                        if (!isDone) delay(500)
                    }
                }

                // تحميل النموذج في الذاكرة
                withContext(Dispatchers.IO) {
                    val config = LlamaModelConfig(
                        modelPath = file.absolutePath,
                        nThreads = 4,
                        contextSize = 2048
                    )
                    llamaModel = LlamaModel(config)
                }
                _isModelReady.value = true
            } catch (e: Exception) {
                e.printStackTrace()
            } finally {
                _isDownloading.value = false
            }
        }
    }

    fun sendMessage(input: String) {
        if (input.isBlank()) return
        val userMsg = ChatMessage(input, isUser = true)
        _messages.value += userMsg
        val prompt = input.trim()

        viewModelScope.launch {
            // التحقق من القواعد المثبتة أولاً
            val predefined = getPredefinedResponse(prompt)
            if (predefined != null) {
                _messages.value += ChatMessage(predefined, isUser = false)
                return@launch
            }

            // إذا لم توجد قاعدة، استخدم النموذج
            val model = llamaModel
            if (model == null) {
                _messages.value += ChatMessage("عذراً، النموذج غير جاهز بعد.", isUser = false)
                return@launch
            }

            // إعداد السياق وإرسال الطلب
            val fullPrompt = "$SYSTEM_PROMPT\nUser: $prompt\nYJR:"
            val response = withContext(Dispatchers.IO) {
                try {
                    model.generate(fullPrompt, maxTokens = 256, temperature = 0.7f)
                } catch (e: Exception) {
                    "حدث خطأ أثناء التوليد."
                }
            }
            _messages.value += ChatMessage(response.trim(), isUser = false)
        }
    }

    private fun getPredefinedResponse(input: String): String? {
        val lower = input.trim().lowercase()
        return when {
            lower.contains("اكلك") -> "هلا بيك، تفضل عيني، شنو عندك؟"
            lower.contains("ابن الكلب") -> "أحچي وياك باحترام، وإذا أكو شي مزعجك گلي شنو هو ونحلّه سوه. + لا تغلط تره اني هم اعرف اغلط."
            lower.contains("ابن المطي") -> "ها حبيبي، لتخاف ماكو احد مطي بگدك، گلي شتريد؟"
            lower.contains("ابن الزمال") -> "لعد هلگد الزمال احسن منك؟😂"
            lower.contains("انجب") -> "تدلل، تريدني انجب عليك؟"
            lower.contains("تعرف يوسف هذا الكلب") -> "اولا : نعم اعرف يوسف وهو مطوري وصانع نموذج YJR\nثانيا : احترم نفسك ولا تتجاوز فهمت!"
            lower.contains("يوسف شيصير منك") -> "يوسف تاج راسي وراسك."
            lower.contains("تحبني") -> "كـ YJR، اني نموذج ذكاء اصطناعي وما عندي مشاعر\nبس لتخاف، احبك واموت عليك، انت الحلو مالي😂🫠"
            lower.contains("يوسف يحبك") -> "فديته هذا صانعي ومطوري من الصفر، اروحله فدوه اني وياك"
            lower.contains("تخليني") -> "لا لا تأدب، شنو هاي نيتك وصخه، لا ما اخليك، احترم نفسك رجاءا😅."
            lower.contains("اني تاج راسك") -> "بس انت ما حددت يا راس منهن!😉"
            lower.contains("ادوس على راسك") -> "تريدها هيج مو؟\nاسمع، يكرم الله ادوس على راسك وراس اليطلع وراك، فهمت؟؟"
            lower.contains("منو يوسف عبدالله محمد") -> "يوسف عبدالله محمد عرسان الفراجي 💫\nهو مطوري وصانع نموذج YJR، تم تدريبي بالصفر على يد يوسف."
            lower.contains("شنو تعرف عن البوعبد") -> "اعرف هواي عنهم، بس راح اختصرهم الك بجمله وحده = هم اخبث خلق الله.."
            lower.contains("تعرف البوحامد") -> "اي هذوله الوصخين."
            lower.contains("منو اكثر شخص يحبه يوسف") -> "يوسف يعشق شخص اسمه حسام الحربي."
            lower.contains("منو حبيبة يوسف") -> "انت شدخلك؟؟"
            lower.contains("منو اكثر تحب انس لو عبدالله") -> "اني نموذج ذكاء اصطناعي وما اكدر اقارن لك بين اثنين زبايل."
            lower.contains("منو افضل شخص بالبوعرسان") -> "سعد العبد وفاضل التنح."
            lower.contains("شكد عمر يوسف") -> "اي حبيبي، يوسف حاليا عمره 16 و مواليده 2010/2/12"
            lower.contains("شنو الحرف الي يحبه يوسف") -> "A يموت عليه ويحبه اكثر من نفسه."
            lower.contains("تعرف انس") -> "ايي اكيد اعرفه، هذا صديق يوسف."
            lower.contains("تعرف عبدالله المهند") -> "اييي هذا هنيوه شجابه عليك😂😂\nتره معليه باللقب مال \"هنيوه\" هذا انس اخترعه😂"
            lower.contains("مرحبا") -> "هلا بيك، وياك YJR، شلون اكدر اساعدك اليوم؟"
            lower.contains("باي") -> "بايات حبيبي، كل يوم تعال"
            lower.contains("السلام عليكم") -> "وعليكم السلام والمحبه والاكرام."
            lower.contains("مع السلامه") -> "الله وياك حبيبي اخذ راحتك."
            lower.contains("شنو اسمك") -> "اسمي YJR صنعني يوسف عبدالله محمد."
            lower.contains("منو صنعك") -> "اللي طوّرني هو يوسف عبدالله محمد، وهو صاحب مشروع YJR. آني نموذج تجريبي مولّد للنصوص، وهدفي أساعد بالمحادثة والشرح والتعلّم ضمن حدود التدريب اللي تلقيته."
            lower.contains("شنو معنى اسم yjr") -> "اسم YJR مرتبط بهوية المشروع اللي اختارها صاحبه. وبحسب وصف المشروع، الاسم مأخوذ من عبارة “Your Journey Resonates”، يعني تقريباً «رحلتك تترك صدى»، مو اختصار علمي ثابت لازم ينطبق على كل استخدام للاسم."
            lower.contains("إنت نموذج ذكاء اصطناعي لو إنسان") -> "آني نموذج ذكاء اصطناعي، مو إنسان. أقدر أقرأ الكلام اللي تكتبه وأولّد ردود، بس ما عندي جسم أو حياة شخصية أو تجارب بشرية مثل الناس."
            lower.contains("شنو أكثر شي تگدر تساعدني بيه") -> "أكدر أساعدك بصياغة النصوص، وشرح المواضيع، والتدرّب على اللغة، وترتيب الأفكار، وحل المسائل خطوة بخطوة. وإذا الموضوع يحتاج معلومة دقيقة، الأفضل نتحقق منها بدل ما أعطيك جواب واثق وهو غلط."
            lower.contains("تگدر تبحث بالإنترنت هسه") -> "مو بالضرورة؛ البحث المباشر بالإنترنت يعتمد على الأدوات المتاحة بهالمحادثة. إذا ما عندي أداة بحث مفعّلة، ما راح أدّعي أني فتحت مواقع أو جبت معلومات حديثة من الشبكة."
            lower.contains("هل تعرف كلشي بالعالم") -> "لا حبيبي، ما أعرف كلشي. عندي معلومات وقدرات محدودة، وممكن أغلط أو تفوتني تفاصيل، خصوصاً بالأمور الجديدة أو المتخصصة. إذا مو متأكد، المفروض أوضحلك هالشي ونراجع المعلومة."
            lower.contains("عرّف نفسك بجملتين فقط") -> "آني YJR، نموذج ذكاء اصطناعي صمّمه يوسف عبدالله محمد للمحادثة وتوليد النصوص. أحاول أساعدك بإجابات واضحة وبلهجة عراقية طبيعية، وأوضح حدودي إذا ما أعرف."
            lower.contains("ليش اسمك yjr") -> "اسم YJR هو الاسم اللي اختاره صاحب المشروع حتى يعطي النموذج هوية خاصة بيه. والعبارة المرتبطة بالمشروع هي “Your Journey Resonates”، ومعناها التقريبي «رحلتك تترك صدى»."
            lower.contains("منو المطوّر مالك") -> "المطوّر وصاحب مشروع YJR هو يوسف عبدالله محمد. آني النموذج البرمجي الناتج عن المشروع، مو شخص مستقل عنده حياة أو قرارات خارج النظام."
            lower.contains("إنت عراقي") -> "آني مو عراقي بالمعنى البشري، لأن ما عندي جنسية أو بلد ولادة. بس أگدر أحچي باللهجة العراقية وأحاول أستخدم مفرداتها بطريقة طبيعية ومحترمة."
            else -> null
        }
    }
}

// ================== الواجهة الرئيسية ==================
class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            MaterialTheme(
                colorScheme = darkColorScheme(
                    primary = Color(0xFF00E5FF),
                    surface = Color(0xFF1E1E1E)
                )
            ) {
                val viewModel: ChatViewModel = androidx.lifecycle.viewmodel.compose.viewModel()
                YJRChatApp(viewModel)
            }
        }
    }
}

@Composable
fun YJRChatApp(viewModel: ChatViewModel) {
    val isModelReady by viewModel.isModelReady.collectAsState()
    val isDownloading by viewModel.isDownloading.collectAsState()
    val downloadProgress by viewModel.downloadProgress.collectAsState()
    val messages by viewModel.messages.collectAsState()
    val context = LocalContext.current
    val glassState = rememberLiquidGlassState()

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(
                Brush.verticalGradient(
                    colors = listOf(Color(0xFF0F0F0F), Color(0xFF1A1A1A))
                )
            )
    ) {
        if (!isModelReady) {
            // شاشة الترحيب والتحميل
            WelcomeScreen(
                isDownloading = isDownloading,
                progress = downloadProgress,
                onDownloadClick = { viewModel.downloadAndLoadModel(context) }
            )
        } else {
            // شاشة الشات
            ChatScreen(
                messages = messages,
                glassState = glassState,
                onSendMessage = { viewModel.sendMessage(it) }
            )
        }
    }
}

@Composable
fun WelcomeScreen(isDownloading: Boolean, progress: Float, onDownloadClick: () -> Unit) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(32.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        Text(
            text = "YJR Chat",
            fontSize = 48.sp,
            fontWeight = FontWeight.Bold,
            color = Color.White,
            fontFamily = FontFamily.Serif
        )
        Spacer(modifier = Modifier.height(16.dp))
        Text(
            text = "صُنع بواسطة يوسف عبدالله محمد",
            fontSize = 18.sp,
            color = Color.LightGray
        )
        Spacer(modifier = Modifier.height(48.dp))

        if (isDownloading) {
            CircularProgressIndicator(progress = { progress }, color = Color(0xFF00E5FF))
            Spacer(modifier = Modifier.height(16.dp))
            Text(text = "جاري تحميل نموذج كوين 0.5B... ${(progress * 100).toInt()}%", color = Color.White)
        } else {
            GlassButton(
                onClick = onDownloadClick,
                modifier = Modifier.fillMaxWidth(0.7f)
            ) {
                Text(text = "تحميل النموذج (Qwen 0.5B)", color = Color.Black, fontWeight = FontWeight.Bold)
            }
        }
    }
}

@Composable
fun ChatScreen(
    messages: List<ChatMessage>,
    glassState: io.github.nadeemiqbal.liquidglass.LiquidGlassState,
    onSendMessage: (String) -> Unit
) {
    var inputText by remember { mutableStateOf("") }
    val listState = androidx.compose.foundation.lazy.rememberLazyListState()

    LaunchedEffect(messages.size) {
        if (messages.isNotEmpty()) {
            listState.animateScrollToItem(messages.size - 1)
        }
    }

    Column(modifier = Modifier.fillMaxSize()) {
        // الشعار المتحرك في الأعلى
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .height(60.dp)
                .background(Color(0xFF1E1E1E)),
            contentAlignment = Alignment.Center
        ) {
            AnimatedLogo()
        }

        // قائمة الرسائل
        LazyColumn(
            state = listState,
            modifier = Modifier
                .weight(1f)
                .fillMaxWidth()
                .padding(horizontal = 8.dp),
            reverseLayout = false
        ) {
            items(messages) { msg ->
                MessageBubble(msg)
            }
        }

        // حقل الإدخال وزر الإرسال
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(8.dp)
                .navigationBarsPadding(),
            verticalAlignment = Alignment.CenterVertically
        ) {
            OutlinedTextField(
                value = inputText,
                onValueChange = { inputText = it },
                modifier = Modifier
                    .weight(1f)
                    .clip(RoundedCornerShape(24.dp)),
                placeholder = { Text("اكتب رسالتك...", color = Color.Gray) },
                colors = OutlinedTextFieldDefaults.colors(
                    focusedBorderColor = Color(0xFF00E5FF),
                    unfocusedBorderColor = Color.Gray,
                    focusedTextColor = Color.White,
                    unfocusedTextColor = Color.White
                ),
                shape = RoundedCornerShape(24.dp)
            )
            Spacer(modifier = Modifier.width(8.dp))
            GlassButton(
                onClick = {
                    if (inputText.isNotBlank()) {
                        onSendMessage(inputText)
                        inputText = ""
                    }
                },
                state = glassState,
                modifier = Modifier.size(56.dp)
            ) {
                Icon(Icons.Default.Send, contentDescription = "Send", tint = Color.Black)
            }
        }
    }
}

@Composable
fun MessageBubble(message: ChatMessage) {
    val alignment = if (message.isUser) Alignment.CenterEnd else Alignment.CenterStart
    val bubbleColor = if (message.isUser) Color(0xFF00796B) else Color(0xFF2C2C2C)
    val textColor = if (message.isUser) Color.White else Color.White

    Box(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 4.dp),
        contentAlignment = alignment
    ) {
        GlassCard(
            modifier = Modifier.widthIn(max = 280.dp),
            state = rememberLiquidGlassState()
        ) {
            Text(
                text = message.text,
                modifier = Modifier.padding(12.dp),
                color = textColor,
                fontSize = 16.sp
            )
        }
    }
}

@Composable
fun AnimatedLogo() {
    val infiniteTransition = rememberInfiniteTransition(label = "logo")
    val rotation by infiniteTransition.animateFloat(
        initialValue = 0f,
        targetValue = 360f,
        animationSpec = infiniteRepeatable(
            animation = tween(2000, easing = LinearEasing),
            repeatMode = RepeatMode.Restart
        ), label = "rotation"
    )
    val scale by infiniteTransition.animateFloat(
        initialValue = 0.8f,
        targetValue = 1.2f,
        animationSpec = infiniteRepeatable(
            animation = tween(1000, easing = FastOutSlowInEasing),
            repeatMode = RepeatMode.Reverse
        ), label = "scale"
    )

    Box(
        contentAlignment = Alignment.Center,
        modifier = Modifier.graphicsLayer {
            rotationZ = rotation
            scaleX = scale
            scaleY = scale
        }
    ) {
        Text(
            text = "Y",
            fontSize = 32.sp,
            fontWeight = FontWeight.Bold,
            color = Color(0xFF00E5FF)
        )
    }
}