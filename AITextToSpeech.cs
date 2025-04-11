using UnityEngine;
using Meta.WitAi.TTS.Integrations;
using Meta.WitAi.TTS.Utilities;
using System.Collections;
using System.Collections.Generic;
using System.Text.RegularExpressions;

public class AITextToSpeech : MonoBehaviour
{
    public static AITextToSpeech Instance { get; private set; } // ����ģʽ����ֹ�ظ�����
    public TTSSpeaker speaker;
    public int maxCharactersPerSegment = 150; // ���ø�С�����ƣ�ȷ���ȶ�
    private bool isSpeaking = false;
    private Coroutine currentSpeakCoroutine = null;

    private void Awake()
    {
        if (Instance == null)
        {
            Instance = this;
            DontDestroyOnLoad(gameObject);
        }
        else
        {
            Destroy(gameObject);
        }
    }

    private void OnDestroy()
    {
        StopAllCoroutines(); // �ͷ�����Э�̣���ֹ Unity ���ص��µ� GCHandle ����
        if (speaker != null)
        {
            speaker.Stop();  // ȷ���ͷ� TTS ��Դ
            speaker = null;
        }
        isSpeaking = false;
        Debug.Log("[AITextToSpeech] Cleaned up TTS resources.");
    }

    private void OnDisable()
    {
        StopAllCoroutines();
        if (speaker != null)
        {
            speaker.Stop();
        }
        isSpeaking = false;
    }

    // �ʶ� AI ���ɵ��ı�
    public void SpeakAIResponse(string aiResponse)
    {
        if (string.IsNullOrEmpty(aiResponse))
        {
            Debug.LogWarning("[AITextToSpeech] Empty response, ignoring.");
            return;
        }

        Debug.Log("[AITextToSpeech] AI Response length: " + aiResponse.Length);

        // ֹͣ��ǰ���ڽ��е��ʶ�
        if (isSpeaking)
        {
            StopCurrentSpeaking();
        }

        if (speaker != null)
        {
            currentSpeakCoroutine = StartCoroutine(DelayedSpeak(aiResponse, 0.3f));
        }
        else
        {
            Debug.LogError("[AITextToSpeech] TTSSpeaker is not assigned!");
        }
    }

    // ֹͣ��ǰ��������
    private void StopCurrentSpeaking()
    {
        if (currentSpeakCoroutine != null)
        {
            StopCoroutine(currentSpeakCoroutine);
            currentSpeakCoroutine = null;
        }

        if (speaker != null && speaker.IsSpeaking)
        {
            speaker.Stop();
        }

        isSpeaking = false;
        Debug.Log("[AITextToSpeech] Stopped previous speaking");
    }

    private IEnumerator DelayedSpeak(string aiResponse, float delay)
    {
        yield return new WaitForSeconds(delay);

        isSpeaking = true;

        if (aiResponse.Length <= maxCharactersPerSegment)
        {
            yield return StartCoroutine(SpeakSegment(aiResponse));
        }
        else
        {
            yield return StartCoroutine(SpeakAISegments(aiResponse));
        }

        isSpeaking = false;
        currentSpeakCoroutine = null;
    }

    private IEnumerator SpeakAISegments(string aiResponse)
    {
        List<string> segments = SplitTextBySentence(aiResponse, maxCharactersPerSegment);
        Debug.Log($"[AITextToSpeech] Total segments: {segments.Count}");

        int segmentIndex = 0;

        foreach (string segment in segments)
        {
            segmentIndex++;
            Debug.Log($"[AITextToSpeech] Speaking segment {segmentIndex}/{segments.Count}: {segment}");

            if (!string.IsNullOrWhiteSpace(segment))
            {
                yield return StartCoroutine(SpeakSegment(segment));
                yield return new WaitForSeconds(0.3f); // Ƭ�μ����ͣ��
            }
        }

        Debug.Log("[AITextToSpeech] All segments completed");
    }

    private IEnumerator SpeakSegment(string segment)
    {
        if (speaker == null)
        {
            Debug.LogError("[AITextToSpeech] TTSSpeaker is missing!");
            yield break;
        }

        speaker.Speak(segment);

        yield return StartCoroutine(WaitForSpeakingToComplete(segment));
    }

    private IEnumerator WaitForSpeakingToComplete(string text)
    {
        float startWaitTime = 0.5f;
        float startElapsed = 0f;

        while (!speaker.IsSpeaking && startElapsed < startWaitTime)
        {
            startElapsed += Time.deltaTime;
            yield return null;
        }

        if (!speaker.IsSpeaking)
        {
            Debug.Log("[AITextToSpeech] Speaking didn't start, waiting a fixed time");
            yield return new WaitForSeconds(text.Length * 0.1f);
            yield break;
        }

        float waitStartTime = Time.time;
        float maxWaitTime = Mathf.Max(5.0f, text.Length * 0.1f) + 2.0f;

        while (speaker.IsSpeaking)
        {
            if (Time.time - waitStartTime > maxWaitTime)
            {
                Debug.Log("[AITextToSpeech] Wait timeout reached, stopping");
                speaker.Stop();
                break;
            }
            yield return null;
        }

        yield return new WaitForSeconds(0.5f);
    }

    private List<string> SplitTextBySentence(string text, int maxLength)
    {
        List<string> segments = new List<string>();

        // ���ӽ�����ǵ�������ʽ (���, �ʺ�, ��̾�ź���ո�����)
        string pattern = @"(?<=[.!?])\s+|(?<=[������])|$";
        string[] sentences = Regex.Split(text, pattern);

        string currentSegment = "";

        foreach (string sentence in sentences)
        {
            if (string.IsNullOrWhiteSpace(sentence))
                continue;

            string trimmedSentence = sentence.Trim();

            // **ȷ�����ᳬ�� 280 �ַ�**
            if (trimmedSentence.Length > 240)
            {
                // ���� 200 �ַ����а�ȫ���
                List<string> sentenceFragments = SplitLongSentence(trimmedSentence, 200);
                segments.AddRange(sentenceFragments);
            }
            else if (currentSegment.Length + trimmedSentence.Length > 150) // Ԥ�� 80 �ַ��ռ�
            {
                segments.Add(currentSegment);
                currentSegment = trimmedSentence;
            }
            else
            {
                if (currentSegment.Length > 0)
                    currentSegment += " ";
                currentSegment += trimmedSentence;
            }
        }

        if (!string.IsNullOrWhiteSpace(currentSegment))
        {
            segments.Add(currentSegment);
        }

        return segments;
    }

    private List<string> SplitLongSentence(string sentence, int maxLength)
    {
        List<string> fragments = new List<string>();
        string[] parts = Regex.Split(sentence, @"(?<=[,;����])\s*");
        string currentFragment = "";

        foreach (string part in parts)
        {
            if (string.IsNullOrWhiteSpace(part)) continue;
            string trimmedPart = part.Trim();

            if (trimmedPart.Length > maxLength)
            {
                if (!string.IsNullOrWhiteSpace(currentFragment))
                {
                    fragments.Add(currentFragment);
                    currentFragment = "";
                }

                for (int i = 0; i < trimmedPart.Length; i += maxLength)
                {
                    fragments.Add(trimmedPart.Substring(i, Mathf.Min(maxLength, trimmedPart.Length - i)));
                }
            }
            else if (currentFragment.Length + trimmedPart.Length > maxLength)
            {
                fragments.Add(currentFragment);
                currentFragment = trimmedPart;
            }
            else
            {
                if (currentFragment.Length > 0) currentFragment += " ";
                currentFragment += trimmedPart;
            }
        }

        if (!string.IsNullOrWhiteSpace(currentFragment)) fragments.Add(currentFragment);
        return fragments;
    }
}
