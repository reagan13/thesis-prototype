import { useParams } from "react-router-dom";
import { Bar } from "react-chartjs-2";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from "chart.js";
import ChartDataLabels from "chartjs-plugin-datalabels";
import { useData } from "../context/DataContext";
import { useState } from "react";

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ChartDataLabels
);

const ChatAnalysis = () => {
  const { id } = useParams();
  const { data } = useData();
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [selectedChartType, setSelectedChartType] = useState(null);

  const activeChat = data.chatHistory.find(
    (chat) => chat.id === data.activeChatId
  );
  if (!activeChat) {
    return <div className="p-6 text-center">No active chat found.</div>;
  }

  const message = activeChat.messages.find((msg) => msg.id === Number(id));
  if (!message || !message.botResponse) {
    return <div className="p-6 text-center">No message found.</div>;
  }

  const intentChartData = {
    labels: [
      message.botResponse.predictions.baseline?.classification.intent?.label ||
        "Unknown",
      message.botResponse.predictions[
        message.botResponse.modelUsed.toLowerCase()
      ]?.classification.intent?.label || "Unknown",
    ],
    datasets: [
      {
        data: [
          (message.botResponse.predictions.baseline?.classification.intent
            ?.confidence || 0) * 100,
          (message.botResponse.predictions[
            message.botResponse.modelUsed.toLowerCase()
          ]?.classification.intent?.confidence || 0) * 100,
        ],
        backgroundColor: ["blue", "orange"],
      },
    ],
  };

  const categoryChartData = {
    labels: [
      message.botResponse.predictions.baseline?.classification.category
        ?.label || "Unknown",
      message.botResponse.predictions[
        message.botResponse.modelUsed.toLowerCase()
      ]?.classification.category?.label || "Unknown",
    ],
    datasets: [
      {
        data: [
          (message.botResponse.predictions.baseline?.classification.category
            ?.confidence || 0) * 100,
          (message.botResponse.predictions[
            message.botResponse.modelUsed.toLowerCase()
          ]?.classification.category?.confidence || 0) * 100,
        ],
        backgroundColor: ["blue", "orange"],
      },
    ],
  };

  const baselineNER =
    message.botResponse.predictions.baseline?.classification.ner || [];
  const hybridNER =
    message.botResponse.predictions[message.botResponse.modelUsed.toLowerCase()]
      ?.classification.ner || [];

  const mergedLabels = [];
  const mergedData = [];
  const mergedBackgroundColors = [];

  baselineNER.forEach((entity) => {
    mergedLabels.push(entity.label);
    mergedData.push(entity.confidence * 100);
    mergedBackgroundColors.push("blue");
  });

  hybridNER.forEach((entity) => {
    mergedLabels.push(entity.label);
    mergedData.push(entity.confidence * 100);
    mergedBackgroundColors.push("orange");
  });

  const nerChartData = {
    labels: mergedLabels,
    datasets: [
      {
        data: mergedData,
        backgroundColor: mergedBackgroundColors,
      },
    ],
  };

  const getChartOptions = (chartType) => ({
    responsive: true,
    indexAxis: chartType === "ner" ? "y" : "x",
    plugins: {
      legend: { display: false },
      title: { display: false, text: "Confidence Scores Comparison" },
      tooltip: {
        callbacks: {
          label: function (context) {
            const datasetIndex = context.datasetIndex;
            const dataIndex = context.dataIndex;
            const value = context.raw.toFixed(2);

            if (chartType === "ner") {
              const entity = mergedLabels[dataIndex];
              const color = context.dataset.backgroundColor[dataIndex];
              const label =
                color === "blue" ? "Baseline" : message.botResponse.modelUsed;
              return `${label}: ${value}%`;
            }

            return dataIndex === 0
              ? `Baseline: ${value}%`
              : `${message.botResponse.modelUsed}: ${value}%`;
          },
        },
      },
      datalabels: { display: false },
    },
    scales: {
      y: {
        beginAtZero: true,
        max: 100,
        title: { display: true, text: "Confidence (%)" },
        grid: { display: chartType !== "ner" },
      },
      x: {
        title: {
          display: true,
          text:
            chartType === "intent"
              ? "Intent"
              : chartType === "category"
              ? "Category"
              : "NER",
        },
        grid: { display: chartType === "ner" },
      },
    },
  });

  const openModal = (chartType) => {
    setSelectedChartType(chartType);
    setIsModalOpen(true);
  };

  const closeModal = () => {
    setIsModalOpen(false);
    setSelectedChartType(null);
  };

  const renderChart = (chartType) => {
    switch (chartType) {
      case "intent":
        return (
          <Bar data={intentChartData} options={getChartOptions("intent")} />
        );
      case "category":
        return (
          <Bar data={categoryChartData} options={getChartOptions("category")} />
        );
      case "ner":
        return <Bar data={nerChartData} options={getChartOptions("ner")} />;
      default:
        return null;
    }
  };

  return (
    <div className="p-6 max-h-screen overflow-y-auto scrollbar-hide scroll-hide h-full">
      <h1 className="text-4xl font-bold text-center text-gray-800 mb-8">
        Chat Analysis
      </h1>
      <div className="mb-8 p-6 bg-white rounded-lg border border-black">
        <p className="text-lg text-gray-700">
          <strong>User Query:</strong> {message.text}
        </p>
        <p className="text-lg text-gray-700 mt-2">
          <strong>Classified Input:</strong>{" "}
          {message.botResponse.classified_input}
        </p>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div
          className="p-6 bg-white rounded-lg border border-black cursor-pointer transform transition-transform duration-200 hover:scale-105"
          onClick={() => openModal("intent")}
        >
          <h2 className="text-xl font-semibold text-gray-800 mb-4 text-center border-b border-black">
            Intent
          </h2>
          <p className="text-gray-700">
            <strong>Baseline:</strong>{" "}
            {message.botResponse.predictions.baseline?.classification.intent
              ?.label || "Unknown"}{" "}
            (
            {(
              (message.botResponse.predictions.baseline?.classification.intent
                ?.confidence || 0) * 100
            ).toFixed(2)}
            %)
          </p>
          <p className="text-gray-700">
            <strong>{message.botResponse.modelUsed}:</strong>{" "}
            {message.botResponse.predictions[
              message.botResponse.modelUsed.toLowerCase()
            ]?.classification.intent?.label || "Unknown"}{" "}
            (
            {(
              (message.botResponse.predictions[
                message.botResponse.modelUsed.toLowerCase()
              ]?.classification.intent?.confidence || 0) * 100
            ).toFixed(2)}
            %)
          </p>
        </div>
        <div
          className="p-6 bg-white rounded-lg border border-black cursor-pointer transform transition-transform duration-200 hover:scale-105"
          onClick={() => openModal("category")}
        >
          <h2 className="text-xl font-semibold text-gray-800 mb-4 text-center border-b border-black">
            Category
          </h2>
          <p className="text-gray-700">
            <strong>Baseline:</strong>{" "}
            {message.botResponse.predictions.baseline?.classification.category
              ?.label || "Unknown"}{" "}
            (
            {(
              (message.botResponse.predictions.baseline?.classification.category
                ?.confidence || 0) * 100
            ).toFixed(2)}
            %)
          </p>
          <p className="text-gray-700">
            <strong>{message.botResponse.modelUsed}:</strong>{" "}
            {message.botResponse.predictions[
              message.botResponse.modelUsed.toLowerCase()
            ]?.classification.category?.label || "Unknown"}{" "}
            (
            {(
              (message.botResponse.predictions[
                message.botResponse.modelUsed.toLowerCase()
              ]?.classification.category?.confidence || 0) * 100
            ).toFixed(2)}
            %)
          </p>
        </div>
        <div
          className="p-6 bg-white rounded-lg border border-black cursor-pointer transform transition-transform duration-200 hover:scale-105"
          onClick={() => openModal("ner")}
        >
          <h2 className="text-xl font-semibold text-gray-800 mb-4 text-center border-b border-black">
            NER
          </h2>
          <p className="text-gray-700">
            <strong>Baseline:</strong>{" "}
            {baselineNER.length > 0
              ? baselineNER.map((entity, index) => (
                  <span key={index} className="mr-2">
                    {entity.entity} ({entity.label},{" "}
                    {(entity.confidence * 100).toFixed(2)}%)
                  </span>
                ))
              : "None"}
          </p>
          <p className="text-gray-700">
            <strong>{message.botResponse.modelUsed}:</strong>{" "}
            {hybridNER.length > 0
              ? hybridNER.map((entity, index) => (
                  <span key={index} className="mr-2">
                    {entity.entity} ({entity.label},{" "}
                    {(entity.confidence * 100).toFixed(2)}%)
                  </span>
                ))
              : "None"}
          </p>
        </div>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-6">
        <div className="p-6 bg-white rounded-lg border border-black">
          <h2 className="text-xl font-semibold text-gray-800 mb-4">
            Baseline Response
          </h2>
          <p className="text-gray-700 mb-4 border border-gray-300 p-4 rounded-lg">
            <strong>Generated Text:</strong>{" "}
            {message.botResponse.predictions.baseline?.response || "N/A"}
          </p>
          <div className="border border-gray-300 p-4 rounded-lg">
            <p>
              <strong>Category:</strong>{" "}
              {message.botResponse.predictions.baseline?.classification.category
                ?.label || "Unknown"}{" "}
              (
              {(
                (message.botResponse.predictions.baseline?.classification
                  .category?.confidence || 0) * 100
              ).toFixed(2)}
              %)
            </p>
            <p>
              <strong>Intent:</strong>{" "}
              {message.botResponse.predictions.baseline?.classification.intent
                ?.label || "Unknown"}{" "}
              (
              {(
                (message.botResponse.predictions.baseline?.classification.intent
                  ?.confidence || 0) * 100
              ).toFixed(2)}
              %)
            </p>
            <p>
              <strong>NER:</strong>{" "}
              {baselineNER.length > 0
                ? baselineNER.map((entity, index) => (
                    <span key={index} className="mr-2">
                      {entity.entity} ({entity.label},{" "}
                      {(entity.confidence * 100).toFixed(2)}%)
                    </span>
                  ))
                : "None"}
            </p>
            <p>
              <strong>classification_time:</strong>{" "}
              {message.botResponse.predictions.baseline?.metrics
                .classification_time || "N/A"}
            </p>
            <p>
              <strong>generation_time:</strong>{" "}
              {message.botResponse.predictions.baseline?.metrics
                .generation_time || "N/A"}
            </p>
            <p>
              <strong>memory_usage:</strong>{" "}
              {message.botResponse.predictions.baseline?.metrics.memory_usage ||
                "N/A"}
            </p>
            <p>
              <strong>overall_time:</strong>{" "}
              {message.botResponse.predictions.baseline?.metrics.overall_time ||
                "N/A"}
            </p>
          </div>
        </div>
        <div className="p-6 bg-white rounded-lg border border-black">
          <h2 className="text-xl font-semibold text-gray-800 mb-4">
            {message.botResponse.modelUsed} Response
          </h2>
          <p className="text-gray-700 mb-4 border border-gray-300 p-4 rounded-lg">
            <strong>Generated Text:</strong>{" "}
            {message.botResponse.predictions[
              message.botResponse.modelUsed.toLowerCase()
            ]?.response || "N/A"}
          </p>
          <div className="border border-gray-300 p-4 rounded-lg">
            <p>
              <strong>Category:</strong>{" "}
              {message.botResponse.predictions[
                message.botResponse.modelUsed.toLowerCase()
              ]?.classification.category?.label || "Unknown"}{" "}
              (
              {(
                (message.botResponse.predictions[
                  message.botResponse.modelUsed.toLowerCase()
                ]?.classification.category?.confidence || 0) * 100
              ).toFixed(2)}
              %)
            </p>
            <p>
              <strong>Intent:</strong>{" "}
              {message.botResponse.predictions[
                message.botResponse.modelUsed.toLowerCase()
              ]?.classification.intent?.label || "Unknown"}{" "}
              (
              {(
                (message.botResponse.predictions[
                  message.botResponse.modelUsed.toLowerCase()
                ]?.classification.intent?.confidence || 0) * 100
              ).toFixed(2)}
              %)
            </p>
            <p>
              <strong>NER:</strong>{" "}
              {hybridNER.length > 0
                ? hybridNER.map((entity, index) => (
                    <span key={index} className="mr-2">
                      {entity.entity} ({entity.label},{" "}
                      {(entity.confidence * 100).toFixed(2)}%)
                    </span>
                  ))
                : "None"}
            </p>
            <p>
              <strong>classification_time:</strong>{" "}
              {message.botResponse.predictions[
                message.botResponse.modelUsed.toLowerCase()
              ]?.metrics.classification_time || "N/A"}
            </p>
            <p>
              <strong>generation_time:</strong>{" "}
              {message.botResponse.predictions[
                message.botResponse.modelUsed.toLowerCase()
              ]?.metrics.generation_time || "N/A"}
            </p>
            <p>
              <strong>memory_usage:</strong>{" "}
              {message.botResponse.predictions[
                message.botResponse.modelUsed.toLowerCase()
              ]?.metrics.memory_usage || "N/A"}
            </p>
            <p>
              <strong>overall_time:</strong>{" "}
              {message.botResponse.predictions[
                message.botResponse.modelUsed.toLowerCase()
              ]?.metrics.overall_time || "N/A"}
            </p>
          </div>
        </div>
      </div>
      {isModalOpen && (
        <div className="fixed inset-0 flex items-center justify-center bg-black bg-opacity-50 z-50">
          <div className="bg-white p-6 rounded-lg border border-black w-11/12 md:w-3/4 lg:w-1/2 max-h-full overflow-y-auto">
            <div className="flex justify-end">
              <button
                onClick={closeModal}
                className="text-gray-600 hover:text-gray-800 font-bold"
                style={{ fontSize: "24px" }}
              >
                ×
              </button>
            </div>
            <h2 className="text-2xl font-semibold text-gray-800 mb-4 text-center">
              Confidence Scores
            </h2>
            {renderChart(selectedChartType)}
          </div>
        </div>
      )}
    </div>
  );
};

export default ChatAnalysis;
