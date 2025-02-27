import { useParams } from "react-router-dom";
import { Bar } from "react-chartjs-2"; // Import Bar from react-chartjs-2
import { Chart as ChartJS, CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend } from 'chart.js'; // Import necessary components from chart.js
import ChartDataLabels from 'chartjs-plugin-datalabels'; // Import ChartDataLabels plugin
import { useData } from "../context/DataContext"; // Import useData
import { useState } from "react"; // Import useState

ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend, ChartDataLabels); // Register chart.js components and ChartDataLabels plugin

const ChatAnalysis = () => {
    const { id } = useParams(); // Get the message ID from the URL
    const { data } = useData(); // Access global state
    const [isModalOpen, setIsModalOpen] = useState(false);
    const [selectedBox, setSelectedBox] = useState(null);

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

    const calculateDuration = (startTime, endTime) => {
        const start = new Date(startTime);
        const end = new Date(endTime);
        const durationMs = end - start;
        const seconds = Math.floor(durationMs / 1000);
        return `${seconds} seconds`;
    };

    const intentChartData = {
        labels: [
            
            message.botResponse.predictions?.baseline?.baseline_predictions?.intent?.label || "Unknown",
            
            message.botResponse.predictions?.hybrid?.hybrid_predictions?.intent?.label || "Unknown",
            
        ],
        datasets: [
            {
               
                data: [
                    
                    (message.botResponse.predictions?.baseline?.baseline_predictions?.intent?.confidence || 0) * 100,
                    (message.botResponse.predictions?.hybrid?.hybrid_predictions?.intent?.confidence || 0) * 100,
                    
                ],
                backgroundColor: ['blue', 'orange'],
            },
        ],
    };
    
    const categoryChartData = {
        labels: [ 
            message.botResponse.predictions?.baseline?.baseline_predictions?.category?.label || "Unknown",
            message.botResponse.predictions?.hybrid?.hybrid_predictions?.category?.label || "Unknown",
        ],
        datasets: [
            {
                
                data: [
                    
                    (message.botResponse.predictions?.baseline?.baseline_predictions?.category?.confidence || 0) * 100,
                    (message.botResponse.predictions?.hybrid?.hybrid_predictions?.category?.confidence || 0) * 100,
                ],
                backgroundColor: ['blue', 'orange'],
            },
        ],
    };

    const getChartOptions = (chartType) => ({
        responsive: true,
        plugins: {
            legend: { display: false },
            title: { display: false, text: 'Confidence Scores Comparison' },
            tooltip: {
                callbacks: {
                    label: function (context) {
                        const datasetIndex = context.datasetIndex;
                        const dataIndex = context.dataIndex;
                        const value = context.raw.toFixed(2);
    
                        return dataIndex === 0 ? `Baseline: ${value}%` : `Hybrid: ${value}%`;
                    },
                },
            },
            datalabels: {
                display: false, // Disable datalabels
            },
        },
        scales: {
            y: { beginAtZero: true, max: 100, title: { display: true, text: 'Confidence (%)' } },
            x: { 
                title: { 
                    display: true, 
                    text: chartType === "intent" ? "Intent" : "Category" 
                } 
            },
        },
    });
    
    

    return (
        <div className="p-6 max-h-screen overflow-y-auto scrollbar-hide scroll-hide h-full">
            <h1 className="text-4xl font-bold text-center text-gray-800 mb-8">Chat Analysis</h1>
            <div className="mb-8 p-6 bg-white rounded-lg shadow-md">
                <p className="text-lg text-gray-700">
                    <strong>User Query:</strong> {message.text}
                </p>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="p-6 bg-white rounded-lg shadow-md">
                    <h2 className="text-xl font-semibold text-gray-800 mb-4">Baseline Response</h2>
                    <p className="text-gray-700 mb-4"><strong>Generated Text:</strong> {message.botResponse.predictions?.baseline?.generated_text || "N/A"}</p>
                    <p>
							<strong>Category:</strong>{" "}
							{message.botResponse.predictions?.baseline?.baseline_predictions
								?.category?.label ||
								message.botResponse.predictions?.baseline?.baseline_predictions
									?.category?.prediction ||
								"Unknown"}{" "}
							(Confidence:{" "}
							{(
								message.botResponse.predictions?.baseline?.baseline_predictions
									?.category?.confidence * 100
							).toFixed(2)}
							%)
						</p>
						<p>
							<strong>Intent:</strong>{" "}
							{message.botResponse.predictions?.baseline?.baseline_predictions
								?.intent?.label ||
								message.botResponse.predictions?.baseline?.baseline_predictions
									?.intent?.prediction ||
								"Unknown"}{" "}
							(Confidence:{" "}
							{(
								message.botResponse.predictions?.baseline?.baseline_predictions
									?.intent?.confidence * 100
							).toFixed(2)}
							%)
						</p>
						<p>
							<strong>NER:</strong>{" "}
							{Array.isArray(
								message.botResponse.predictions?.baseline?.baseline_predictions
									?.ner
							) &&
							message.botResponse.predictions?.baseline?.baseline_predictions
								?.ner.length > 0
								? message.botResponse.predictions?.baseline?.baseline_predictions?.ner.map(
										(entity, index) => (
											<span key={index} className="mr-2">
												{"Text: "}
												{entity.text},{" Type: "}
												{entity.type} (Confidence:{" "}
												{(entity.confidence * 100).toFixed(2)}%)
											</span>
										)
								  )
								: "None"}
						</p>
                    <p>
                        <strong>Weighted Sum:</strong>{" "}
                        {message.botResponse.predictions?.baseline?.weighted_sum.toFixed(4)}
                    </p>
                    <p>
                        <strong>Processing Time:</strong>{" "}
                        {calculateDuration(
                            message.botResponse.predictions?.baseline?.start_time,
                            message.botResponse.predictions?.baseline?.end_time
                        )}
                    </p>
                </div>
                <div className="p-6 bg-white rounded-lg shadow-md">
                    <h2 className="text-xl font-semibold text-gray-800 mb-4">Hybrid Response</h2>
                    <p className="text-gray-700 mb-4"><strong>Generated Text:</strong> {message.botResponse.predictions?.hybrid?.generated_text || "N/A"}</p>
                    <p>
							<strong>Category:</strong>{" "}
							{message.botResponse.predictions?.hybrid?.hybrid_predictions
								?.category?.label ||
								message.botResponse.predictions?.hybrid?.hybrid_predictions
									?.category?.prediction ||
								"Unknown"}{" "}
							(Confidence:{" "}
							{(
								message.botResponse.predictions?.hybrid?.hybrid_predictions
									?.category?.confidence * 100
							).toFixed(2)}
							%)
						</p>
						<p>
							<strong>Intent:</strong>{" "}
							{message.botResponse.predictions?.hybrid?.hybrid_predictions
								?.intent?.label ||
								message.botResponse.predictions?.hybrid?.hybrid_predictions
									?.intent?.prediction ||
								"Unknown"}{" "}
							(Confidence:{" "}
							{(
								message.botResponse.predictions?.hybrid?.hybrid_predictions
									?.intent?.confidence * 100
							).toFixed(2)}
							%)
						</p>
						<p>
							<strong>NER:</strong>{" "}
							{Array.isArray(
								message.botResponse.predictions?.hybrid?.hybrid_predictions?.ner
							) &&
							message.botResponse.predictions?.hybrid?.hybrid_predictions?.ner
								.length > 0
								? message.botResponse.predictions?.hybrid?.hybrid_predictions?.ner.map(
										(entity, index) => (
											<span key={index} className="mr-2">
												{"Text: "}
												{entity.text},{" Type: "}
												{entity.type} (Confidence:{" "}
												{(entity.confidence * 100).toFixed(2)}%)
											</span>
										)
								  )
								: "None"}
						</p>
                    <p>
                        <strong>Weighted Sum:</strong>{" "}
                        {message.botResponse.predictions?.hybrid?.weighted_sum.toFixed(4)}
                    </p>
                    <p>
                        <strong>Processing Time:</strong>{" "}
                        {calculateDuration(
                            message.botResponse.predictions?.hybrid?.start_time,
                            message.botResponse.predictions?.hybrid?.end_time
                        )}
                    </p>
                </div>
            </div>
            <div className="p-6 bg-white rounded-lg shadow-md mt-6">
                <h2 className="text-xl font-semibold text-gray-800 mb-4 text-center">Category Confidence Score</h2>
                <Bar data={categoryChartData} options={getChartOptions("category")} />
            </div>
            <div className="p-6 bg-white rounded-lg shadow-md mt-6">
                <h2 className="text-xl font-semibold text-gray-800 mb-4 text-center">Intent Confidence Score</h2>
                <Bar data={intentChartData} options={getChartOptions("intent")} />
            </div>
            
        </div>
    );
};

export default ChatAnalysis;